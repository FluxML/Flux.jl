# Data: stream a text dataset from the HuggingFace Hub, tokenize it with GPT-2's BPE, and pack the
# token ids into rows of `block_size + 1` tokens. The rows are written once to a dataset cached in
# `data_dir`, which `datasets` memory-maps; later runs load it from there.

using HuggingFaceDatasets, HuggingFaceTokenizers, PythonCall

# Token ids are 1-based in Julia, but 0-based in the tokenizer and in the cached datasets, which
# are shared with the PyTorch port. The conversions are all in this file: `tokenize`,
# `detokenize`, `tokenize_batch` and `getbatch`.
const EOT = 50257   # <|endoftext|>, prepended to every document

# The Hub dataset and, for each split, the Hub split and the documents taken from it. FineWeb-Edu
# has only a train split: its first 100k documents (~100M tokens) are the validation set. The
# training documents are selected with `filter` rather than `skip(100_000)`, which in datasets 4,
# where Reactant's Python requirements hold this environment, skips only 1000 documents.
const DATASETS = Dict(
    "fineweb" => (; path=("HuggingFaceFW/fineweb-edu", "sample-10BT"),
                    train=("train", s -> filter((b, i) -> i .>= 100_000, s; batched=true, with_indices=true)),
                    val=("train", s -> s.take(100_000))),
    "tinystories" => (; path=("roneneldan/TinyStories",),
                        train=("train", identity), val=("validation", identity)),
)

gpt2_tokenizer() = from_pretrained(Tokenizer, "openai-community/gpt2")

tokenize(tok, text::AbstractString) = encode(tok, text).ids .+ 1
detokenize(tok, ids::AbstractVector{<:Integer}) = decode(tok, ids .- 1)

# The 0-based ids of a batch of documents, each prefixed with <|endoftext|>, as one numpy array.
# `encode_batch_fast` runs in parallel in Rust, and scales well only with large batches. The ids
# are collected in Python, so that no Julia wrapper keeps a per-document `Encoding` alive.
const ENCODE_BATCH = pyeval("""lambda tok, texts, eot, np: np.concatenate(
    [np.asarray([eot] + e.ids, dtype=np.uint16) for e in tok.encode_batch_fast(texts)])""", @__MODULE__)

function tokenize_batch(tok, texts)
    arr = ENCODE_BATCH(tok.py_tokenizer, texts, EOT - 1, pyimport("numpy"))
    ids = pyconvert(Vector{UInt16}, arr)
    PythonCall.pydel!(arr)
    return ids
end

# Until HuggingFaceDatasets.jl wraps `Dataset.from_generator` (JuliaGenAI/HuggingFaceDatasets.jl#109).
const ITER = pyeval("lambda ds: iter(ds)", @__MODULE__)

# The `split` (`:train` or `:val`) of `dataset` ("fineweb" or "tinystories"), as rows of
# `block_size + 1` token ids. The documents are streamed, tokenized, and concatenated and cut into
# rows within each batch of 10k documents, whose last partial row is dropped (~0.01% of the
# tokens). The first call writes the rows to `data_dir/<dataset>/`; later calls load them.
function token_blocks(dataset, split; block_size, data_dir)
    spec = DATASETS[dataset]
    hfsplit, select = spec[split]
    stream = select(load_dataset(spec.path...; split=hfsplit, streaming=true).select_columns("text"))
    n, tok = block_size + 1, gpt2_tokenizer()
    blocks = map(stream; batched=true, batch_size=10_000, remove_columns=["text"]) do b
        ids = tokenize_batch(tok, b["text"])
        Dict("ids" => reshape(ids[1:length(ids) ÷ n * n], n, :))
    end
    ds = HuggingFaceDatasets.datasets.Dataset.from_generator(ITER; gen_kwargs=pydict(ds=blocks.py),
             fingerprint="$dataset-$split-gpt2-$n", cache_dir=joinpath(data_dir, dataset))
    return with_format(Dataset(ds), "julia")
end

# Rows `i:i+B-1` as inputs and next-token targets (1-based ids, block_size × B), and the first row
# of the next batch. Wraps around to the first row at the end.
function getbatch(ds, i, B)
    i + B - 1 > length(ds) && (i = 1)
    ids = Int32.(ds[i:i+B-1]["ids"]) .+ Int32(1)
    return (ids[1:end-1, :], ids[2:end, :]), i + B
end
