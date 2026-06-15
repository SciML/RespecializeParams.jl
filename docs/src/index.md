# RespecializeParams.jl

RespecializeParams.jl provides type-stable opaque parameter containers for SciML
solvers. The goal is to keep `typeof(p)` uniform across underlying payload
types — so a precompiled solver path is shared — while still recovering the
concrete payload type inside `f` for fully specialized inner kernels.

There are two containers:

  - [`OpaqueParams`](@ref) — for `isbits` payloads. Backed by `Vector{UInt8}`;
    `unpack` is a single `unsafe_load` with no allocation.
  - [`OpaqueRef`](@ref) — for non-`isbits` payloads. Backed by `Ref{Any}`;
    `unpack` is a pointer load with a `::T` assertion. Identity-preserving:
    mutating the underlying object is observable through the wrapper.

## Installation

```julia
using Pkg
Pkg.add("RespecializeParams")
```

## Why?

When you call `solve(prob, alg)`, the solver specializes on `typeof(prob.p)`.
Different parameter struct types produce different precompiled solver
specializations. If you want the same solver code path to be reused across
several parameter struct variants — but still want `f` to use the concrete
parameter type for full specialization of the inner kernel — you need a way
to make `p` look uniform on the outside while still carrying the payload type
information that `f` needs.

RespecializeParams gives you exactly that: a wrapper whose Julia type is fixed,
and an in-`f` unpack that recovers the concrete payload type without allocation.

## Quick start (ODE example)

```julia
using RespecializeParams, OrdinaryDiffEqTsit5

struct LorenzP
    σ::Float64
    ρ::Float64
    β::Float64
end

# Carry the payload type on the rhs (as a type parameter), not on p.
struct OpaqueRHS{T, F} <: Function
    f::F
end
OpaqueRHS(::Type{T}, f::F) where {T, F} = OpaqueRHS{T, F}(f)

@inline function (r::OpaqueRHS{T})(du, u, op::OpaqueParams, t) where {T}
    p = unpack(op, T)
    r.f(du, u, p, t)
    return nothing
end

function lorenz_kernel!(du, u, p::LorenzP, t)
    du[1] = p.σ * (u[2] - u[1])
    du[2] = u[1] * (p.ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - p.β * u[3]
    return nothing
end

prob = ODEProblem(
    OpaqueRHS(LorenzP, lorenz_kernel!),
    [1.0, 0.0, 0.0],
    (0.0, 5.0),
    pack(LorenzP(10.0, 28.0, 8 / 3)),
)

solve(prob, Tsit5())
```

`typeof(prob.p) === OpaqueParams` regardless of which concrete payload type was
packed, so the solver hits the same precompiled code path across all such
problems.

## API

### `OpaqueParams` (isbits payloads)

```@docs
OpaqueParams
pack
unpack(::OpaqueParams, ::Type{T}) where {T}
unsafe_unpack
unpack_checked(::OpaqueParams, ::Type{T}) where {T}
repack!(::OpaqueParams, ::T) where {T}
```

### `OpaqueRef` (any payload)

```@docs
OpaqueRef
pack_any
unpack(::OpaqueRef, ::Type{T}) where {T}
unpack_checked(::OpaqueRef, ::Type{T}) where {T}
repack!(::OpaqueRef, ::T) where {T}
```

## Choosing between `OpaqueParams` and `OpaqueRef`

| Property                     | `OpaqueParams`         | `OpaqueRef`             |
|:---------------------------- |:---------------------- |:----------------------- |
| Payload constraint           | `isbits` only          | anything                |
| Storage                      | `Vector{UInt8}` (copy) | `Ref{Any}` (by ref)     |
| Mutation propagation         | no (snapshot at pack)  | yes (identity preserved)|
| `unpack` cost                | `unsafe_load`          | pointer load + tag check|
| Best for                     | plain numeric structs  | structs with `Vector`/`Dict`/`Function` fields |

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.
  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:

      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
