## Description

This repository demonstrates a Physics-Informed AI/ML framework for thermodynamic simulation and optimization. The framework integrates physics-based modeling with neural networks to accelerate simulations while maintaining engineering fidelity.

This Physics-Informed AI/ML approach is extensible to modeling and optimization of other electric vehicle components, such as battery thermal management systems, or HVAC subsystems. Unlike traditional methods that provide only scalar outputs (e.g., COP), this framework predicts full-field variables such as temperature, enthalpy, and flow distribution across the component. This allows detailed analysis and design optimization for complex thermal systems in EVs.

## Test Case: Condenser 

As a test case, a condenser was modeled to study heat transfer and fluid dynamics within the refrigeration cycle. Neural networks are trained to predict temperature fields in the fluid domain, enabling rapid evaluation of condenser performance under different operating conditions. Validation datasets are incorporated to ensure accuracy and reliability of the predictions.
The figure below illustrates the geometry of the condenser.

![photo_2026-01-04_15-32-16](https://github.com/user-attachments/assets/b66479d1-d553-49ab-9f70-dc6b53057803)
                                                                          *Side view of the Condenser generated using Python:*
<p align="center">
<img width="782" height="269" alt="Figure_1" src="https://github.com/user-attachments/assets/c333e6a1-bb89-44ed-9cff-fc66736e6b76" />


In the heat pump cycle of an EV, the condenser releases heat to the surroundings and converts the refrigerant from vapor to liquid. Accurate modeling is critical for evaluating the system’s thermal performance. In this project:

- The condenser geometry and flow domain are defined based on a secondary fluid A/C and refrigeration system.

- Physics-informed neural networks are trained to predict temperatures and enthalpy in the fluid domain.

- Validation datasets are generated to compare AI/ML predictions against conventional simulation results (ANSYS FLUENT).

This approach allows high-fidelity simulation of the condenser without requiring full computational fluid dynamics (CFD) runs for every operating point. 


### Main output

The figures below compare the Physics-Informed AI/ML predictions with reference CFD (ANSYS FLUENT) results for validation.  

<p align="center">
  <img width="477" height="487" alt="AI-CFD(1)" src="https://github.com/user-attachments/assets/07a65180-aa12-4f8c-b52f-1f1f56d1c330" />
  <img width="307" height="307" alt="AI-CFD(2)" src="https://github.com/user-attachments/assets/47c995d9-e0b9-4a8c-afc0-f158b896a643" />
<a>



The models show strong agreement on key thermal predictions, as both the CFD and the AI/ML model accurately capture the same outlet and maximum temperatures. While the visualization of vortex dynamics differs, the models are not contradictory. The primary distinction lies in how they represent the flow: the CFD shows complex, unsteady vortex shedding, while the AI model depicts a simplified, stable vortex, effectively showing a time-averaged result. 

For detailed information, please refer to the [CFD](CFD) and [AI/ML](AI-ML) folders of this project.



## Benefits of the Approach

- Reduced Simulation Time: Physics-informed ML drastically reduces computational cost compared to full CFD.

- High Accuracy: Incorporates physical laws and validation datasets to maintain reliability.

- Scalable: The trained models can be applied to different operating conditions and integrated into larger EV thermal management studies.

- Decision Support: Enables OEM-level design and optimization of heat pump.


## Simulation Workflow

1- Geometry Setup: The condenser and surrounding fluid domain are defined.

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

## From Source

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


## License

PhysicsNeMo is provided under the Apache License 2.0. Please see [LICENSE.txt](./LICENSE.txt)
for the full license text. Enterprise SLA, support, and preview access are available
under NVAIE.
