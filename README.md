# Description

This repository contains our implementation for learning prescriptions related to refractive vision problems (myopia, presbyopia, astigmatism etc.).

Clone the entire repository and navigate to the root directory.

```shell
git clone --recurse-submodules git@github.com:complight/learned_prescription.git
cd learned_prescription
```


# (0) Install the required dependencies

`requirements.txt` can help you to install the required packages using `pip`:

```shell
pip3 install -r requirements.txt
```


# (1) Running the optimization

Once you have the requirements successfully installed, you are ready to run the optimisation.

```shell
python3 main.py
```

You can also adjust the parameters used in the optimization routine by passing arguments. To learn more about parameters:

```shell
python3 main.py --help
```

Here is a sample syntax that adjusts several key parameters of our optimizations:

```shell
python3 main.py --device cuda --filename dataset/lenna.png --directory sample --backlight read
```
