.. image:: https://readthedocs.org/projects/megamix/badge/?version=latest
    :target: http://megamix.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://travis-ci.org/14thibea/megamix.svg?branch=master
    :target: https://travis-ci.org/14thibea/megamix
    :alt: Build Status on Travis
    
.. image:: https://badge.fury.io/py/megamix.svg
    :target: https://badge.fury.io/py/megamix
    :alt: version available on PyPI
   
=======
MeGaMix
=======

.. highlight:: bash

The MeGaMix **python package** provides several **clustering models** like k-Means and other Gaussian Mixture Models.

Installation
------------

Create the conda environment and install required dependencies :

```bash
conda create -n megamix python=3.8 && conda activate megamix
cat requirements.txt | xargs pip install
python setup.py install
```

Install CPC dependencies :

```bash
git clone https://github.com/MarvinLvn/CPC_audio.git
cd CPC_audio
sed -i 's/name: cpc37/name: megamix/g' environment.yml
conda env update -f environment.yml
python setup.py develop
```

Documentation
-------------

See the complete documentation `online <http://megamix.readthedocs.io/en/latest/>`_

1) Extract the features
_______________________

```bash
python speech/utils/extract_mfccs.py --db /private/home/marvinlvn/DATA/CPC_data/train/English_LibriVox_extracted_full_random/8h/8h_nb_0 \
--out /private/home/marvinlvn/DATA/CPC_data/train/English_LibriVox_extracted_full_random/mfccs_8h_nb_0.pt \
--type mfcc
```


Test
----


The megamix package comes with a unit-tests suit. To run it, first install *pytest* on your Python environment::

  $ pip install pytest

Then run the tests with::

  $ pytest