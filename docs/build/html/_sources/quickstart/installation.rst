Installation
=============

To install MEGPrep, you need to ensure that Nextflow and Docker are installed on your system. Please follow these steps for installation:

Install Nextflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install Nextflow using the following command:

.. code-block:: bash

    $ curl -s https://get.nextflow.io | bash
    $ chmod +x nextflow
    $ mkdir -p $HOME/.local/bin/
    $ mv nextflow $HOME/.local/bin/
    $ nextflow info # Confirm that Nextflow is installed correctly


Install Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Docker according to your operating system. For detailed installation instructions, please visit the `Docker official website <https://docs.docker.com/get-docker/>`_.

.. code-block:: bash
    $ docker info # Confirm that docker is installed correctly


`Install Singularity <https://docs.sylabs.io/guides/3.5/user-guide/index.html>`_ [options]


Download MEGPrep Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker pull megprep:latest



