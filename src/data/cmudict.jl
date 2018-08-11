module CMUDict

export cmudict

using ..Data: deps

const version = "0.7b"
const cache_prefix = "https://cache.julialang.org"

function load()
  suffixes = ["", ".phones", ".symbols"]
  if isdir(deps("cmudict"))
    if all(isfile(deps("cmudict", "cmudict$x")) for x in suffixes)
      return
    end
  end
  @info "Downloading CMUDict dataset"
  mkpath(deps("cmudict"))
  for x in suffixes
    download("$cache_prefix/http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-$version$x",
             deps("cmudict", "cmudict$x"))
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
