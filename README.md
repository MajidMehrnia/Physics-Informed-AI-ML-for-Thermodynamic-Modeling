## Description

This repository demonstrates a Physics-Informed AI/ML framework for thermodynamic simulation and optimization. The framework integrates physics-based modeling with neural networks to accelerate simulations while maintaining engineering fidelity.

This Physics-Informed AI/ML approach is extensible to modeling and optimization of other electric vehicle components, such as battery thermal management systems, or HVAC subsystems. Unlike traditional methods that provide only scalar outputs (e.g., COP), this framework predicts full-field variables such as temperature, enthalpy, and flow distribution across the component. This allows detailed analysis and design optimization for complex thermal systems in EVs.

As a test case, a Thermostatic Expansion Valve (TXV) is modeled to study heat transfer and fluid dynamics within the refrigeration cycle. Neural networks are trained to predict temperature fields in the fluid domain, enabling rapid evaluation of TXV performance under different operating conditions. Validation datasets are incorporated to ensure accuracy and reliability of the predictions.
The figure below illustrates the geometry of the TXV

![txv3](https://github.com/user-attachments/assets/087ddbbf-d97f-4fc8-95a1-7708da448bd4)

### Main output

The figure below compares the AI/ML predictions with reference CFD results for validation. The TXV model and the Physics-Informed AI/ML predictions were validated against reference CFD simulations. 

![T](https://github.com/user-attachments/assets/fc61ba71-062c-471a-ad14-0d883693f221)

The models show strong agreement on key thermal predictions, as both the CFD and the AI/ML model accurately capture the same outlet and minimum temperatures. While the visualization of vortex dynamics differs, the models are not contradictory. The primary distinction lies in how they represent the flow: the CFD shows complex, unsteady vortex shedding, while the AI model depicts a simplified, stable vortex, effectively showing a time-averaged result.



## Benefits of the Approach

- Reduced Simulation Time: Physics-informed ML drastically reduces computational cost compared to full CFD.

- High Accuracy: Incorporates physical laws and validation datasets to maintain reliability.

- Scalable: The trained models can be applied to different operating conditions and integrated into larger EV thermal management studies.

- Decision Support: Enables OEM-level design and optimization of heat pump.

## Test Case: Thermostatic Expansion Valve (TXV)

The TXV regulates refrigerant flow in the heat pump cycle of an electric vehicle. Accurate modeling is critical for evaluating the system’s thermal performance. In this project:

- The TXV geometry and flow domain are defined based on standard automotive designs.

- Physics-informed neural networks are trained to predict temperatures and enthalpy in the fluid domain.

- Validation datasets are generated to compare AI/ML predictions against conventional simulation results (ANSYS FLUENT).

This approach allows high-fidelity simulation of the TXV without requiring full computational fluid dynamics (CFD) runs for every operating point. 


## Simulation Workflow

1- Geometry Setup: The TXV and surrounding fluid domain are defined.

2- Physics-Based Modeling: Governing equations for mass, momentum, and energy are implemented to capture thermodynamic behavior.

3- Neural Network Training: AI/ML models are trained on simulated or experimental data to predict fluid temperature fields.

4- Validation: AI/ML predictions are validated against reference CFD or analytical results.

5- Integration: Trained models can be integrated with larger EV thermal management simulations to assess heat pump cycle performance efficiently.

## Getting Started

To run the simulations and train the AI/ML models:

1- Clone the repository.

2- Install dependencies listed in requirements.

3- Follow the step-by-step workflow scripts in the scripts.

4- Check the results for post-processed temperature and performance data.

The NVIDIA open-source Physics-ML framework is employed in this work. NVIDIA PhysicsNeMo is an open-source deep-learning framework for building, training, fine-tuning, and inferring Physics AI models using state-of-the-art SciML methods for AI4Science and engineering.

PhysicsNeMo provides Python modules to compose scalable and optimized training and inference pipelines to explore, develop, validate, and deploy AI models that combine physics knowledge with data, enabling real-time predictions.

Whether you are exploring the use of neural operators, GNNs, or transformers, or are
interested in Physics-Informed Neural Networks or a hybrid approach in between, PhysicsNeMo
provides you with an optimized stack that will enable you to train your models at scale.


- [More About PhysicsNeMo](#more-about-physicsnemo)
  - [Scalable GPU-Optimized Training Library](#scalable-gpu-optimized-training-library)
  - [A Suite of Physics-Informed ML Models](#a-suite-of-physics-informed-ml-models)
  - [Seamless PyTorch Integration](#seamless-pytorch-integration)
  - [Easy Customization and Extension](#easy-customization-and-extension)
  - [AI4Science Library](#ai4science-library)
    - [Domain-Specific Packages](#domain-specific-packages)
- [Who is Using and Contributing to PhysicsNeMo](#who-is-using-and-contributing-to-physicsnemo)
- [Why Use PhysicsNeMo](#why-are-they-using-physicsnemo)
- [Getting Started](#getting-started-with-physicsnemo)
- [Resources](#resources)
- [Installation](#installation)
- [Contributing](#contributing-to-physicsnemo)
- [Communication](#communication)
- [License](#license)

<!-- tocstop -->

## More About PhysicsNeMo

At a granular level, PhysicsNeMo is developed as modular functionality and therefore
provides built-in composable modules that are packaged into a few key components:

<!-- markdownlint-disable -->
Component | Description |
---- | --- |
[**physicsnemo.models**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.models.html) | A collection of optimized, customizable, and easy-to-use families of model architectures such as Neural Operators, Graph Neural Networks, Diffusion models, Transformer models, and many more|
[**physicsnemo.datapipes**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.datapipes.html) | Optimized and scalable built-in data pipelines fine-tuned to handle engineering and scientific data structures like point clouds, meshes, etc.|
[**physicsnemo.distributed**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.html) | A distributed computing sub-module built on top of `torch.distributed` to enable parallel training with just a few steps|
[**physicsnemo.curator**](https://github.com/NVIDIA/physicsnemo-curator) | A sub-module to streamline and accelerate the process of data curation for engineering datasets|
[**physicsnemo.sym.geometry**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/csg_and_tessellated_module.html) | A sub-module to handle geometry for DL training using Constructive Solid Geometry modeling and CAD files in STL format|
[**physicsnemo.sym.eq**](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/nodes.html) | A sub-module to use PDEs in your DL training with several implementations of commonly observed equations and easy ways for customization|
<!-- markdownlint-enable -->

For a complete list, refer to the PhysicsNeMo API documentation for
[PhysicsNeMo](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html).

## AI4Science Library

Usually, PhysicsNeMo is used either as:

- A complementary tool to PyTorch when exploring AI for SciML and AI4Science applications.
- A deep learning research platform that provides scale and optimal performance on
NVIDIA GPUs.

### Domain-Specific Packages

The following are packages dedicated to domain experts of specific communities, catering
to their unique exploration needs:

- [PhysicsNeMo CFD](https://github.com/NVIDIA/physicsnemo-cfd): Inference sub-module of PhysicsNeMo
  to enable CFD domain experts to explore, experiment, and validate using pretrained
  AI models for CFD use cases.
- [PhysicsNeMo Curator](https://github.com/NVIDIA/physicsnemo-curator): Inference sub-module
  of PhysicsNeMo to streamline and accelerate the process of data curation for engineering
  datasets.
- [Earth-2 Studio](https://github.com/NVIDIA/earth2studio): Inference sub-module of PhysicsNeMo
  to enable climate researchers and scientists to explore and experiment with pretrained
  AI models for weather and climate.

### Scalable GPU-Optimized Training Library

PhysicsNeMo provides a highly optimized and scalable training library for maximizing the
power of NVIDIA GPUs.
[Distributed computing](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.html)
utilities allow for efficient scaling from a single GPU to multi-node GPU clusters with
a few lines of code, ensuring that large-scale
physics-informed machine learning (ML) models can be trained quickly and effectively.
The framework includes support for advanced
[optimization utilities](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.utils.html#module-physicsnemo.utils.capture),
[tailor-made datapipes](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.datapipes.html),
and [validation utilities](https://github.com/NVIDIA/physicsnemo-sym/tree/main/physicsnemo/sym/eq)
to enhance end-to-end training speed.

### A Suite of Physics-Informed ML Models

PhysicsNeMo offers a library of state-of-the-art models specifically designed
for Physics-ML applications. Users can build any model architecture by using the underlying
PyTorch layers and combining them with curated PhysicsNeMo layers.

The [Model Zoo](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.models.html#model-zoo)
includes optimized implementations of families of model architectures such as
Neural Operators:

- [Fourier Neural Operators (FNOs)](physicsnemo/models/fno)
- [DeepONet](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/neural_operators/deeponet.html)
- [DoMINO](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/external_aerodynamics/domino/readme.html)
- [Graph Neural Networks (GNNs)](physicsnemo/models/gnn_layers)
- [MeshGraphNet](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/vortex_shedding_mgn/readme.html)
- [MeshGraphNet for Lagrangian](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/lagrangian_mgn/readme.html)
- [XAeroNet](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/cfd/external_aerodynamics/xaeronet/readme.html)
- [Diffusion Models](physicsnemo/models/diffusion)
- [Correction Diffusion Model](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/generative/corrdiff/readme.html)
- [DDPM](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/generative/diffusion/readme.html)
- [PhysicsNeMo GraphCast](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/examples/weather/graphcast/readme.html)
- [Transsolver](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/darcy_transolver)
- [RNNs](https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models)
- [SwinVRNN](https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models/swinvrnn)
- [Physics-Informed Neural Networks (PINNs)](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html)

And many others.

These models are optimized for various physics domains, such as computational fluid
dynamics, structural mechanics, and electromagnetics. Users can download, customize, and
build upon these models to suit their specific needs, significantly reducing the time
required to develop high-fidelity simulations.

### Seamless PyTorch Integration

PhysicsNeMo is built on top of PyTorch, providing a familiar and user-friendly experience
for those already proficient with PyTorch.
This includes a simple Python interface and modular design, making it easy to use
PhysicsNeMo with existing PyTorch workflows.
Users can leverage the extensive PyTorch ecosystem, including its libraries and tools,
while benefiting from PhysicsNeMo's specialized capabilities for physics-ML. This seamless
integration ensures users can quickly adopt PhysicsNeMo without a steep learning curve.

For more information, refer to [Converting PyTorch Models to PhysicsNeMo Models](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.models.html#converting-pytorch-models-to-physicsnemo-models).

### Easy Customization and Extension

PhysicsNeMo is designed to be highly extensible, allowing users to add new functionality
with minimal effort. The framework provides Pythonic APIs for
defining new physics models, geometries, and constraints, making it easy to extend its
capabilities to new use cases.
The adaptability of PhysicsNeMo is further enhanced by key features such as
[ONNX support](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.deploy.html)
for flexible model deployment,
robust [logging utilities](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.launch.logging.html)
for streamlined error handling,
and efficient
[checkpointing](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.launch.utils.html#module-physicsnemo.launch.utils.checkpoint)
to simplify model loading and saving.

This extensibility ensures that PhysicsNeMo can adapt to the evolving needs of researchers
and engineers, facilitating the development of innovative solutions in the field of physics-ML.

Detailed information on features and capabilities can be found in the [PhysicsNeMo documentation](https://docs.nvidia.com/physicsnemo/index.html#core).

[Reference samples](examples/README.md) cover a broad spectrum of physics-constrained
and data-driven
workflows to suit the diversity of use cases in the science and engineering disciplines.

> [!TIP]
> Have questions about how PhysicsNeMo can assist you? Try our [Experimental] chatbot,
> [PhysicsNeMo Guide](https://chatgpt.com/g/g-PXrBv20SC-modulus-guide), for answers.


## Learning AI Physics

- [Explore Jupyter Notebooks on Hugging Face](https://huggingface.co/collections/nvidia/physicsnemo)
- [AI4Science PhysicsNeMo Bootcamp](https://github.com/openhackathons-org/End-to-End-AI-for-Science)
- [Self-Paced DLI Training](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-04+V1)
- [Deep Learning for Science and Engineering Lecture Series](https://www.nvidia.com/en-us/on-demand/deep-learning-for-science-and-engineering/)
- [Video Tutorials](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=physicsnemo&sort=relevance&sortDir=desc)

## Resources

- [Getting Started Webinar](https://www.nvidia.com/en-us/on-demand/session/gtc24-dlit61460/?playlistId=playList-bd07f4dc-1397-4783-a959-65cec79aa985)
- [PhysicsNeMo: Purpose and Usage](https://www.nvidia.com/en-us/on-demand/session/dliteachingkit-setk5002/)
- [AI4Science PhysicsNeMo Bootcamp](https://github.com/openhackathons-org/End-to-End-AI-for-Science)
- [PhysicsNeMo Pretrained Models](https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=PhysicsNeMo&page=&pageSize=)
- [PhysicsNeMo Datasets and Supplementary Materials](https://catalog.ngc.nvidia.com/resources?filters=&orderBy=scoreDESC&query=PhysicsNeMo&page=&pageSize=)

## Installation

The following instructions help you install the base PhysicsNeMo modules to get started.
There are additional optional dependencies for specific models that are listed under
[optional dependencies](#optional-dependencies).
The training recipes are not packaged into the pip wheels or the container to keep the
footprint low. We recommend users clone the appropriate training recipes and use them
as a starting point. These training recipes may require additional example-specific dependencies,
as indicated through their associated `requirements.txt` file.

### PyPI

The recommended method for installing the latest version of PhysicsNeMo is using PyPI:

```Bash
pip install nvidia-physicsnemo
python -c "import physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__)"
```

The installation can also be verified by running the [Hello World](#hello-world) example.

#### Optional Dependencies

PhysicsNeMo has many optional dependencies that are used in specific components.
When using pip, all dependencies used in PhysicsNeMo can be installed with
`pip install nvidia-physicsnemo[all]`. If you are developing PhysicsNeMo, developer dependencies
can be installed using `pip install nvidia-physicsnemo[dev]`. Otherwise, additional dependencies
can be installed on a case-by-case basis. Detailed information on installing the
optional dependencies can be found in the
[Getting Started Guide](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html).

### NVCR Container

The recommended PhysicsNeMo Docker image can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo)
(refer to the NGC registry for the latest tag):

```Bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.06
```

Inside the container, you can clone the PhysicsNeMo git repositories and get
started with the examples. The command below shows the instructions to launch
the PhysicsNeMo container and run examples from this repo:

```bash
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia \
--rm -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/cfd/darcy_fno/
pip install warp-lang # install NVIDIA Warp to run the Darcy example
python train_fno_darcy.py
```

## From Source

### Package

For a local build of the PhysicsNeMo Python package from source, use:

```Bash
git clone git@github.com:NVIDIA/physicsnemo.git && cd physicsnemo

pip install --upgrade pip
pip install .
python -c "import physicsnemo; print('PhysicsNeMo version:', physicsnemo.__version__)"
```

### Source Container

To build the PhysicsNeMo Docker image:

```bash
docker build -t physicsnemo:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target deploy -f Dockerfile .
```

Alternatively, you can run `make container-deploy`.

To build the CI image:

```bash
docker build -t physicsnemo:ci \
    --build-arg TARGETPLATFORM=linux/amd64 --target ci -f Dockerfile .
```

Alternatively, you can run `make container-ci`.

Currently, only `linux/amd64` and `linux/arm64` platforms are supported. If using
`linux/arm64`, some dependencies like `warp-lang` might not install correctly.

## PhysicsNeMo Migration Guide

NVIDIA Modulus has been renamed to NVIDIA PhysicsNeMo. For migration:

- Use `pip install nvidia-physicsnemo` rather than `pip install nvidia-modulus`
  for PyPI wheels.
- Use `nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>` rather than
  `nvcr.io/nvidia/modulus/modulus:<tag>` for Docker containers.
- Replace `nvidia-modulus` with `nvidia-physicsnemo` in your pip requirements
  files (`requirements.txt`, `setup.py`, `setup.cfg`, `pyproject.toml`, etc.).
- In your code, change the import statements from `import modulus` to
  `import physicsnemo`.

The old PyPI registry and the NGC container registry will be deprecated soon
and will not receive any bug fixes/updates. The old checkpoints will remain
compatible with these updates.

More details to follow soon.

## DGL to PyTorch Geometric Migration Guide

PhysicsNeMo supports a wide range of Graph Neural Networks (GNNs),
including MeshGraphNet and others.
Currently, PhysicsNeMo uses the DGL library as its GNN backend,
with plans to completely transition to PyTorch Geometric (PyG) in a future release.
For more details, please refer to the [DGL-to-PyG migration guide](https://github.com/NVIDIA/physicsnemo/blob/main/examples/dgl_to_pyg_migration.md).

## Contributing to PhysicsNeMo

PhysicsNeMo is an open-source collaboration, and its success is rooted in community
contributions to further the field of Physics-ML. Thank you for contributing to the
project so others can build on top of your contributions.

For guidance on contributing to PhysicsNeMo, please refer to the
[contributing guidelines](CONTRIBUTING.md).

## Cite PhysicsNeMo

If PhysicsNeMo helped your research and you would like to cite it, please refer to the [guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff).

## Communication

- GitHub Discussions: Discuss new architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- PhysicsNeMo Forum: The [PhysicsNeMo Forum](https://forums.developer.nvidia.com/t/welcome-to-the-physicsnemo-ml-model-framework-forum/178556)
hosts an audience of new to moderate-level users and developers for general chat, online
discussions, collaboration, etc.

## Feedback

Want to suggest some improvements to PhysicsNeMo? Use our [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

PhysicsNeMo is provided under the Apache License 2.0. Please see [LICENSE.txt](./LICENSE.txt)
for the full license text. Enterprise SLA, support, and preview access are available
under NVAIE.
