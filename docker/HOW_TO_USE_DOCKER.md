## Docker Image for Building/Testing/Running Face Tracking Project(telef)

2018/06/21

# Prerequisite

* Get docker image `.tar.gz` file (from mango Users/yucheol/telef)
* Install `nvidia-docker`
* Get the source code for this project from [git](https://github.com/ycjungSubhuman/telef/)
* Follow "Adiitional Setup" part in README.md (You can find the file in mango Users/yucheol/telef)

# Loading the Image

`docker load < $PATH_TO_YOUR_TAR_IMAGE`

# Running the IDE for This Project

```bash
cd $YOUR_TELEF_PROJECT_FOLDER/docker
./run_clion
```

If you get permission error, try `sudo ./run_clion`

When you launch the script, you will be asked an account for JetBrain. Make one if you need it.

# Running an Arbitrary Command for This Project

```bash
cd $YOUR_TELEF_PROJECT_FOLDER/docker
./telef_run.sh $YOUR_COMMAND
```

## Examples

### Run shell

`./telef_run.sh bash`

## Contact

If you have any problem, contact Yucheol (ycjung@postech.ac.kr)
