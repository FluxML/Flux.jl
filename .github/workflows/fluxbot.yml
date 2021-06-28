name: FluxBot

on:
  issue_comment:
    types: [created, edited]

jobs:
  build:
    if: contains(github.event.comment.body, '@ModelZookeeper')
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        julia-version: [1.5.0]
        julia-arch: [x86]
        os: [ubuntu-latest]
    steps:
      - uses: actions/checkout@af513c7a016048ae468971c52ed77d9562c7c819 # v1.0.0
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - name: Install dependencies
        run: julia --project=.fluxbot/ -e 'using Pkg; Pkg.instantiate()'
      - name: FluxBot.respond
        env:
          FLUXBOT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          BOT_SECRET: ${{ secrets.BOT_SECRET }}
          MODELZOO_TRIGGER_TOKEN: ${{ secrets.MODELZOO_TRIGGER_TOKEN }}
        run: julia --project=.fluxbot -e 'using FluxBot; FluxBot.trial()'
