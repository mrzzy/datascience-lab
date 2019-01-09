# Project Title

🔬Machine Learning Experiments

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure that [Anaconda](https://www.anaconda.com/download/), and [Git](https://git-scm.com/) is downloaded.

### Installing

A step by step series of examples that tell you how to get a development env running

Clone this repository

```
git clone https://github.com/J370/MLlaboratory.git
```

Change into the right directory with ```cd``` command.

Create a conda environment

```
conda create --name fruit tensorflow
```

When conda asks you to proceed, type ```y```

Activate conda environment
```
conda activate fruit
```

Install requirements by first changing to ```fruit_not_fruit``` by using the ```cd``` command.
```
pip install -r requirements.txt
```

If there are any requirement errors when running, you can install the required requirement individually.
```
conda install (requirement)
```

## Running

Use the command
```
python src\client.py
```
to run the programme.

Current working directory is \MLlaboratory\fruit_not_fruit

Refer to above should you face any requirement errors.

## Built With

* [Anaconda](https://www.anaconda.com/) - Packages used
* [Git](https://git-scm.com/) - Cloning and managing of this repository

## Authors

* **mrzzy** - *Initial work* - [mrzzy](https://github.com/mrzzy)
* **J370** - *Edits and changes* - [J370](https://github.com/J370)

See also the list of [contributors](https://github.com/mrzzy/MLlaboratory/graphs/contributors) who participated in this project.
