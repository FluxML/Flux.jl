# Data: stream a text dataset from the HuggingFace Hub, tokenize it with GPT-2's BPE, and store the
# token ids as flat UInt16 files (`train.bin`, `val.bin`) that training memory-maps.

using HuggingFaceDatasets, HuggingFaceTokenizers, PythonCall, Mmap, Printf

const EOT = 50256   # <|endoftext|>, 0-based; prepended to every document

const DATASETS = Dict(
    # FineWeb-Edu has only a train split: its first `val_tokens` tokens become the validation set.
    "fineweb" => (; path="HuggingFaceFW/fineweb-edu", name="sample-10BT", train="train",
                    val=nothing, val_tokens=100_000_000),
    "tinystories" => (; path="roneneldan/TinyStories", name=nothing, train="train",
                        val="validation", val_tokens=0),
)

gpt2_tokenizer() = from_pretrained(Tokenizer, "openai-community/gpt2")

# The "text" column of `split`, streamed in batches of `batch_size` documents. Streaming reads the
# parquet files as it goes, so nothing is converted to Arrow on disk. The batches are iterated in
# Python without formatting, so they stay Python lists of strings that go to the Rust tokenizer
# without a round trip through Julia.
function text_batches(spec, split; batch_size)
    args = spec.name === nothing ? (spec.path,) : (spec.path, spec.name)
    ds = with_format(load_dataset(args...; split, streaming=true), nothing)
    return ds.py.select_columns("text").iter(; batch_size)
end

# The ids of a batch of documents, each prefixed with <|endoftext|>, as one numpy array.
# `encode_batch_fast` runs in parallel in Rust, and scales well only with large batches. The ids
# are collected in Python, so that no Julia wrapper keeps a per-document `Encoding` alive.
const ENCODE_BATCH = pyeval("""lambda tok, texts, eot, np: np.concatenate(
    [np.asarray([eot] + e.ids, dtype=np.uint16) for e in tok.encode_batch_fast(texts)])""", @__MODULE__)

function tokenize_batch(tok, texts::Py)
    arr = ENCODE_BATCH(tok.py_tokenizer, texts, EOT, pyimport("numpy"))
    ids = pyconvert(Vector{UInt16}, arr)
    PythonCall.pydel!(arr)
    return ids
end

# Tokenize `dataset` ("fineweb" or "tinystories") into `data_dir/<dataset>/{train,val}.bin`.
# Each file is written under a temporary name and renamed once complete, so an interrupted run
# leaves no truncated file behind. `max_docs` limits the train split (for quick tests).
function prepare(dataset; data_dir, max_docs=typemax(Int), batch_size=10_000)
    spec = DATASETS[dataset]
    dir = joinpath(data_dir, dataset)
    mkpath(dir)
    tok = gpt2_tokenizer()
    if spec.val === nothing
        write_split(tok, text_batches(spec, spec.train; batch_size), dir;
                    val_tokens=spec.val_tokens, max_docs)
    else
        write_split(tok, text_batches(spec, spec.val; batch_size), dir; val_tokens=typemax(Int))
        write_split(tok, text_batches(spec, spec.train; batch_size), dir; val_tokens=0, max_docs)
    end
    return dir
end

# Write the first `val_tokens` tokens (rounded up to a whole batch) to val.bin and the rest to
# train.bin.
function write_split(tok, batches, dir; val_tokens, max_docs=typemax(Int))
    files = Dict(name => joinpath(dir, name * ".bin") for name in ("val", "train"))
    ios = Dict(name => open(path * ".tmp", "w") for (name, path) in files)
    ntok = Dict("val" => 0, "train" => 0)
    ndocs, t0, last_log = 0, time(), 0
    for batch in batches
        ndocs >= max_docs && break
        texts = batch["text"]
        ids = tokenize_batch(tok, texts)
        name = ntok["val"] < val_tokens ? "val" : "train"
        write(ios[name], ids)
        ntok[name] += length(ids)
        ndocs += length(texts)
        # Python's memory is released only when these wrappers are, and Julia's GC can't see it.
        PythonCall.pydel!(texts)
        PythonCall.pydel!(batch)
        total = ntok["val"] + ntok["train"]
        if total - last_log >= 100_000_000
            last_log = total
            @printf "[prepare] %d docs, %.2fB tokens, %.1fM tok/s\n" ndocs total/1e9 total/(time()-t0)/1e6
            flush(stdout)
        end
    end
    for (name, io) in ios
        close(io)
        ntok[name] > 0 ? mv(files[name] * ".tmp", files[name]; force=true) : rm(files[name] * ".tmp")
    end
    @printf "[prepare] done: %d docs, val %d tokens, train %d tokens, %.0f s\n" ndocs ntok["val"] ntok["train"] time()-t0
    return ntok
end

# Batches of `B` sequences of `T` tokens read sequentially from a token file, with the targets
# shifted by one. Ids are returned 1-based for Julia indexing.
struct TokenLoader
    tokens::Vector{UInt16}
    B::Int
    T::Int
end

TokenLoader(path::AbstractString, B, T) =
    TokenLoader(Mmap.mmap(path, Vector{UInt16}, filesize(path) ÷ sizeof(UInt16)), B, T)

ntokens(l::TokenLoader) = length(l.tokens)

# The batch starting at the 0-based token offset `pos`, and the offset of the next batch.
# Wraps around to the start once the file is exhausted.
function getbatch(l::TokenLoader, pos::Int)
    n = l.B * l.T
    pos + n + 1 > ntokens(l) && (pos = 0)
    buf = Int32.(view(l.tokens, pos+1:pos+n+1)) .+ Int32(1)
    x = reshape(buf[1:n], l.T, l.B)
    y = reshape(buf[2:n+1], l.T, l.B)
    return (x, y), pos + n
end

# `julia --project=. data.jl <dataset> [data_dir]`
if abspath(PROGRAM_FILE) == @__FILE__
    dataset = ARGS[1]
    data_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "data")
    prepare(dataset; data_dir)
end
