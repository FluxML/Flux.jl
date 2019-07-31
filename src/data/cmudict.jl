module CMUDict

export cmudict

using ..Data: deps, download_and_verify

const version = "0.7b"
const cache_prefix = "https://cache.julialang.org"

function load()
  suffixes_and_hashes = [(""       , "209a8b4cd265013e96f4658632a9878103b0c5abf62b50d4ef3ae1be226b29e4"),
                        (".phones" , "ffb588a5e55684723582c7256e1d2f9fadb130011392d9e59237c76e34c2cfd6"),
                        (".symbols", "408ccaae803641c6d7b626b6299949320c2dbca96b2220fd3fb17887b023b027")]
  if isdir(deps("cmudict"))
    if all(isfile(deps("cmudict", "cmudict$x")) for (x, _) in suffixes_and_hashes)
      return
    end
  end
  @info "Downloading CMUDict dataset"
  mkpath(deps("cmudict"))
  for (x, hash) in suffixes_and_hashes
    download_and_verify("$cache_prefix/http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-$version$x",
             deps("cmudict", "cmudict$x"), hash)
  end
end

function phones()
  load()
  Symbol.(first.(split.(split(read(deps("cmudict", "cmudict.phones"),String),
                        "\n", keepempty = false), "\t")))
end

function symbols()
  load()
  Symbol.(split(read(deps("cmudict", "cmudict.symbols"),String),
                "\n", keepempty = false))
end

function rawdict()
  load()
  Dict(String(xs[1]) => Symbol.(xs[2:end]) for xs in
       filter(!isempty, split.(split(read(deps("cmudict", "cmudict"),String), "\n"))))
end

validword(s) = isascii(s) && occursin(r"^[\w\-\.]+$", s)

cmudict() = filter(p -> validword(p.first), rawdict())

alphabet() = ['A':'Z'..., '0':'9'..., '_', '-', '.']

end
