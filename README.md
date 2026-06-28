# Trajectory_Augmentation
- **Institute/College Name** - *International Institute of Information Technology Bangalore (IIIT B)*, *26/C, Hosur Rd, Electronic City Phase I, Electronic City, Konappana Agrahara, Karnataka 560100*
- **Final Presentation Google Drive Link** - https://drive.google.com/file/d/1mUA5i8B6y7bWLt_Lu_dnNAC5M6ztGBtl/view?usp=sharing

Note: The PDF linked above is also uploaded to the repository.
- **Full Submission Demo Video Link** - https://youtu.be/TiQOQ-mVkQI

- **Setup & Result Reproducibility Video Link** - https://youtu.be/t5jbTETiGAY

Note: The reproducibility video was shot using a screenshare but the fast moving frames in the IDE were not able to be captured because of lower-end device. We have saved the maximum output that we could of the terminal that was being run during the reproducibility. 

### Project Artefacts

- **Technical Documentation** - Create a **docs** folder and add all technical details in markdown files inside this folder explaining the project Technical Stack, List of OSS libraries/projects used along with their links, the technical architecture of your solution, implementation details, installation instructions, user guide, salient features of the projects. Kindly add screenshots wherever possible.

Note: This can be found at `docs/`
- **[Important]** Create a file `docs/ax.md` whiere you explain in detail how you utilizes open weight models and/or agentic development tools to implement your solution. Explain in detail your  Agentic AI setup , Agentic workflows, Reasoning & planning pipelines, Tool use / tool chaining, Coding assistants, agents, harness, MCP servers, agents.md, skills, Memory / context handling, Multi-agent orchestration systems, etc. Please highlight from your experience - what worked and **what did not work**.

Note: This can be found at `docs/ax.md`
- **Source Code** - Create a **src** folder and add all developed project source codes (including training & benchmark evaluation codes) in the repo. The code must be capable of being successfully installed/executed and must run consistently on the intended platforms.

Note: This can be found at `src/`

- **Models Used** - Not Applicable
- **Models Published** - Not Applicable
- **Datasets Used** - https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- **Datasets Published** - https://huggingface.co/datasets/jdhr/libero_trajectory_augmented

#### Final Presentation

Unlike Phase 1 presentation, in Phase 2 you can freely decide the template, flow and content of your technical presentation. Ensure you cover all aspects of your solution - innovation, novelty, architecture, open datasets/models developed and used, final deliverable details, KPIs of your solution, AI/Agent use, any other details. 

Note: Please find the PDF for the final presentation in the link given above. The pdf is present in the this repository as well. 

#### Full Submission Demo Video

Create a high quality video demonstration your solution in real life and showcasing how it is actually solves the proposed AX Hackathon problem.

Note: The demo video can be found here - https://youtu.be/TiQOQ-mVkQI

#### Setup & Result Reproducibility Video

To ensure reproducibility of results and to verify the presented KPIs, we require you to create a video demonstrating:
- Step by step project installation,
- Data/model download steps, 
- Execution of all required codes to train the developed models (if any)
- Execution of all evaluation codes to reproduce the presented results/KPIs 

Note: The reproducibility video can be found here - https://youtu.be/t5jbTETiGAY

### Attribution 

This project builds upon the foundations of several incredible open-source robotics and dataset ecosystems. We have directly utilized code, libraries, and datasets from the following repositories:

- LIBERO Benchmark: We utilized the LIBERO codebase for the underlying environment task definitions and the expert HDF5 demonstrations (Hugging Face Datasets).
Original Paper: arXiv:2306.03310

- Hugging Face LeRobot: We utilized the LeRobot ecosystem as the standard schema for Vision-Language-Action (VLA) models and relied on their v3.0 dataset documentation for formatting rules.

- Robosuite: We utilized Robosuite as the underlying MuJoCo physics engine to roll out and render the robotic environments.

- LIBERO-to-LeRobot Converters: We adapted scripts to facilitate the complex state-space mapping from native LIBERO format to the Hugging Face v3.0 schema. Code was referenced and adapted from community converters, including Tavish9/any4lerobot and sainavaneet/libero-to-Lerobot.

### Acknowledgements
This work was supported by the 3D Vision and Language Lab, IIIT-Bangalore. We extend our gratitude for their guidance and support throughout the developement of this project. 