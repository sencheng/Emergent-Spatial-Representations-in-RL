###Code for the paper : Navigation task and action space drive the emergence of egocentric and allocentric spatial representations

Using model-free reinforcement learning to study spatial navigation in an artificial agent using different tasks (Currently - guidance and aiming). 
The network representations are then analysed in order to understand what types of spatial representations are used, and the factors that shape these representations. 

**The modules and architecture used are based on [CoBeL-RL by Walther et al]**


## Requirements
- Software
	-CoBeL-RL framework
- Libraries :
The required libraries can be found in the setup instructions for CoBeL-RL. 

______

In order to test the code, please follow the following steps : 

- Clone the CoBeL-RL framework on to your system
- Follow the instructions to set up CoBeL-RL, including the libraries and software requirements, and check that CoBeL-RL demo runs without issues.
- Clone or download the repository for this project to your system.
- Add the directory for the repository to your PYTHONPATH variable. One way to do this is to add `export PYTHONPATH="${PYTHONPATH}:/path/to/emergent_spatial_representations/"` to your .bashrc profile
- Navigate into the demo folder : `cd demo`
- Run the demo simulations : `python test_image.py` and `python test_vector.py`

## Common issues
If you encounter issues running the demo code, make sure to check the following : 
- Your system uses python3.6.x or newer
- CoBeL-RL demo runs without issues
- Blender executable environment variable is set correctly before running the demo, check CoBeL-RL repository for details (needs to be set each time a new terminal is opened, or in your .bashrc profile)
- Python environment variable is set correctly, or your .bashrc profile is modified to find the python modules for CoBeL and this project.

## Contact
sandhiya.vijayabaskaran@rub.de
