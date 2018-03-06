module CMUDict

export cmudict

using ..Data: deps

const version = "0.7b"
const cache_prefix = "https://cache.julialang.org"

function load()
  suffixes = ["", ".phones", ".symbols"]
  if isdir(deps("cmudict"))
    if all(isfile.(["cmudict$x" for x in suffixes]))
      return
    end
  end
  mkpath(deps("cmudict"))
  for x in suffixes
    download("$cache_prefix/http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-$version$x", deps("cmudict", "cmudict$x"))
  end
end

function phones()
  load()
  Symbol.(first.(split.(split(readstring(deps("cmudict", "cmudict.phones")),
                        "\n", keep = false), "\t")))
end

function symbols()
  load()
  Symbol.(split(readstring(deps("cmudict", "cmudict.symbols")),
                "\n", keep = false))
end

function rawdict()
  load()
  Dict(String(xs[1]) => Symbol.(xs[2:end]) for xs in
       filter(!isempty, split.(split(readstring(deps("cmudict", "cmudict")), "\n"))))
end

validword(s) = ismatch(r"^[\w\-\.]+$", s)

cmudict() = filter((s, ps) -> validword(s), rawdict())

alphabet() = ['A':'Z'..., '0':'9'..., '_', '-', '.']

end
