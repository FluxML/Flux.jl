# Flux Contributor’s Guide

Welcome to the Flux community! 

**Flux.jl** is the de facto machine learning library in the **Julia ecosystem** and one of the foundational Julia packages. It is written in *pure Julia*. This is very different from most other state-of-the-art machine learning libraries which typically leverage languages like C or C++ under the hood.

Flux has always had the mission of being a simple, hackable and performant approach to machine learning, which is extended to a number of scientific domains by means of differentiable programming. 

Julia already provides many of the features that machine learning researchers and practitioners need. However, the ecosystem is still rapidly growing. This provides a great opportunity for you to contribute to the future of machine learning and the Julia language in general.

In this guide, you can find information on how to make your first contribution to Flux, some ideas for getting started, and some guidelines on how to submit a contribution. 

## Why contribute to Flux?

You can make a difference to one of the most quickly growing deep learning frameworks, the Julia Language, and the future of open-source software as a whole. Open source projects rely on contributions from volunteers. Contributions enable both the project and volunteers to grow and develop. No matter how you contribute to Flux, our community is dedicated to ensuring it will be a great experience for you!

Contributing to Flux brings several benefits to you and the community such as:

* Contribute to the cause of open and reproducible science
* Help grow the adoption of the FluxML ecosystem
* Become a member of a community that’s excited about open source and sharing knowledge
* Polish your teaching skills by helping others get started using Flux
* Showcase your work by creating examples that others can use in their projects
* Build a track record of public contributions which will help build your career

## Tools

Before you start contributing to Flux, you need the following tools:

* **Julia:** For more information on how to install Julia, see [Download Julia (in under 2.5 minutes)](https://www.youtube.com/watch?v=t67TGcf4SmM).
* **IDE:** You can set one of the following IDEs as well as their extensions for developing in Julia:
   * [Julia for Visual Studio Code](https://www.julia-vscode.org/) (this is the recommended IDE)
   * [IJulia](https://github.com/JuliaLang/IJulia.jl)
   * [Pluto.jl](https://github.com/fonsp/Pluto.jl)
   * [Julia-vim](https://github.com/JuliaEditorSupport/julia-vim)
   * [Emacs major mode for the Julia programming language](https://github.com/JuliaEditorSupport/julia-emacs)
   * [Julia language support for Notepad++](https://github.com/JuliaEditorSupport/julia-NotepadPlusPlus)
* **Knowledge of Git and how to create a pull request:** For more information on getting started with Git, see [Making a first Julia pull request](https://kshyatt.github.io/post/firstjuliapr/).

## Learn Flux

If you need to learn about Julia and Flux, then you can check out the following resources:

* [JuliaAcademy](https://juliaacademy.com/) introductory courses to Julia and Flux
* Flux’s official [Getting Started](https://fluxml.ai/getting_started.html) tutorial
* [Deep Learning with Flux - A 60 Minute Blitz](https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html): a quick intro to Flux loosely based on PyTorch’s tutorial
* [Flux Model Zoo](https://github.com/FluxML/model-zoo) showcases various demonstrations of models that you can reuse with your own data
* [Flux’s official documentation](https://fluxml.ai/Flux.jl/stable/)

## Get help

The Flux community is more than happy to help you with any questions that you might have related to your contribution. You can get in touch with the community in any of the following channels:

* **Discourse forum:** Go to [Machine Learning in Julia community](https://discourse.julialang.org/c/domain/ml/24)
* **Community team:** Join the group of [community maintainers](https://github.com/FluxML/ML-Coordination-Tracker) supporting the FluxML ecosystem (see the [Zulip stream](https://julialang.zulipchat.com/#narrow/stream/237432-ml-ecosystem-coordination) as well)
* **Slack:** Join the [Official Julia Slack](https://julialang.org/slack/) for casual conversation (see `#flux-bridged` and `#machine-learning`)
* **Zulip:** Join the [Zulip Server for the Julia programming language](https://julialang.zulipchat.com/login/) community
* **Stack Overflow:** Search for [Flux/ ML](https://stackoverflow.com/questions/tagged/flux.jl) tags
* **Events:** Attend the [ML bi-weekly calls](https://julialang.org/community/#events)

## How to contribute to Flux
In this section, you can find a step by step guide to make your first Flux contribution. These are the suggested steps for a complete beginner but feel free to adapt them to your own style and journey.

Before you begin working on your contribution, make sure you have installed Julia and an IDE (see section [Tools](#tools)). In case of any issues, remember you can always [get help](#get-help) from the community. 

To make your first contribution to Flux:

1. Read about getting started with **Git** and **how to contribute to an open-source project**. The tutorial [Making a first Julia pull request](https://kshyatt.github.io/post/firstjuliapr/) is a great starting point for learning about git and how to contribute to the Julia language in general. 
2. Read the [Contribution ideas](#contribution-ideas) section to get inspiration.
3. Set up your local environment. Depending on your contribution, fork and clone the Flux, Model Zoo or Flux website GitHub repositories. 
4. Build Flux locally.
5. Familiarise yourself with the Flux source code. See [Flux source code organization](#flux-source-code-organization) for more information.
6. Read the Contribution guidelines section. Familiarise yourself with the guidelines for filing a bug report and submitting a contribution.

After you make your first contribution, try to find an issue that is more challenging to you. You can always find a way to help expand the Flux ecosystem by creating extensions and new packages.

## Contribution ideas

You can contribute to Flux in several ways but we are also open to any suggestions or ideas that you have. In this section you can find some ideas on how you can contribute to the Flux core functionality, Flux Model Zoo, and tutorials. If you have any other ideas, keep in mind that we are open to contributions in all shapes and forms!

### Flux core functionality

The following types of issues are a great way to get started with contributing to 
core Flux functionality:

#### Help Wanted Issues

One of the best ways to contribute is by looking at issues labelled [help wanted](https://github.com/FluxML/Flux.jl/labels/help%20wanted). These issues are not always very beginner-friendly. However, you are welcome to [ask clarifying questions](#get-help) or just browse help wanted issues to see if there is anything that seems interesting to help with.

#### Good First Issues

Issues with the tag **good first issue** are a great starting point for new contributors. You can find a few ideas listed in the [good first issue](https://github.com/FluxML/Flux.jl/labels/good%20first%20issue) section. As mentioned above, if any of these issues seem interesting but there is no clear next step in your mind, then please feel free to ask for suggestions on getting started. Oftentimes in open source, issues labelled as good first issue actually take some back and forth between maintainers and contributors before a new contributor can tackle the issues.

### Flux Model Zoo

If you are already using Flux, then you can share the models that you have created. The Flux Model Zoo contains various demonstrations of models created using Flux that may freely be used as a starting point for other models. For more information on how to share your models see [Model Zoo's README](https://github.com/FluxML/model-zoo#contributing).

### Write tutorials

Tutorials are an important part of making Deep Learning and Flux accessible to everyone. The [Flux Tutorials](https://fluxml.ai/tutorials.html) section already has some great tutorials but we welcome new tutorials, especially if they are beginner-focused. Keep in mind that beginners tend to write the best tutorials! 

We suggest the following starting points as ideas for writing tutorials: 

* A good first step is to find a tutorial you find helpful from another ecosystem and try to rewrite it using Flux (making sure to properly attribute and not plagiarise).
* Look through the questions people ask on the community forum. Write or expand the ideas (give proper credit to the post author).
* Showcase your own projects. Have you created a model for a new application? You can give a full step by step description of your model and describe its results. 

No matter if other people have already written about the same topic before. You can always bring a new point of view, dig deeper into a topic, or provide a new use case using a different data set. For more information on how to create a pull request with your tutorial, see [Flux website README](https://github.com/FluxML/fluxml.github.io).

### Other

We are open to contributions in all shapes and forms! If you have an idea, please [suggest it](#get-help) to the community and we will do our best to help you bring it to life!

## Contribution guidelines

Flux.jl and its ecosystem follow the ColPrac: Contributor's Guide on Collaborative Practises for Community Packages guide. For more information about best practises on contributing to packages, see the [ColPrac guide](https://github.com/SciML/ColPrac).

We also suggest taking a look at the [Julia Ecosystem Contributor’s Guide](https://julialang.org/contribute/) guide which goes over higher-level topics like why you should contribute to the Julia ecosystem, different contribution pathways, and more!

In this section, you can find information on the following:

* [How to file a bug report](#how-to-file-a-bug-report)
* [How to submit a contribution](#how-to-submit-a-contribution)
* [Flux source code organisation](#flux-source-code-organisation)

### How to file a bug report

Have you found a possible issue? File a bug report and include information about how to reproduce the error. 

Before opening a new GitHub issue, make sure you try the following:

* Search the existing issues or the [Julia Discourse forum](https://discourse.julialang.org/) to see if someone else has already noticed the same problem.
* Do some simple debugging techniques to help isolate the problem.
* Consider running julia-debug with a debugger such as gdb or lldb. Obtaining even a simple backtrace is very useful.

When filing the bug report, provide the following information (if applicable):

* The full error message, including the backtrace.
* A minimal working example. Also, include the smallest chunk of code that triggers the error. Ideally, this should be code that can be pasted into a REPL or run from a source file. If the code is larger than (say) 50 lines, consider putting it in a [gist](https://gist.github.com/).
* The version of Julia as provided by the versioninfo() command. Occasionally, the longer output produced by versioninfo(verbose = true) may be useful also, especially if the issue is related to a specific package.

> **Note:** When pasting code blocks or output, put triple back quotes (```) around the text so GitHub will format it nicely. Code statements should be surrounded by single backquotes (`). Be aware that the @ sign tags users on GitHub, so references to macros should always be in single back quotes. See [GitHub's guide on Markdown](https://guides.github.com/features/mastering-markdown) for more formatting tricks.

### How to submit a contribution

You can use [this template](https://github.com/FluxML/Flux.jl/blob/master/.github/pull_request_template.md) as a starting point when creating a new pull request. 

### Flux source code organization

The following table shows how the Flux code is organized:

| Directory  | Contents |
| ------------- | ------------- |
| docs  | Documentation site  |
| paper  | Paper that describes Flux |
| perf  |    |
| src    |  Source for Flux  |
| test   |  Test suites  |
