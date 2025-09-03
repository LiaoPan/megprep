.. megprep documentation master file, created by
   sphinx-quickstart on Wed Aug 30 22:11:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MEGPrep's documentation!
=================================
`MEGPrep <https://github.com/LiaoPan/megprep>`_ is a fully automated preprocessing pipeline for MEG (Magnetoencephalography) data, built on the MNE-Python framework and leveraging the power of Nextflow.
It is specifically designed to address the challenges of large-scale MEG data processing with a strong emphasis on reproducibility, efficiency, and user-friendliness in various research environments.


Features
---------

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Reliability and Robustness
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            MEGPrep ensures reliable and robust MEG data processing. Standardized environments through containerization, using Docker and Singularity, guarantee consistent results across computational setups. This minimizes variability and ensures reproducibility across different systems, facilitating cross-subject and cross-site studies of MEG data.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Modularity and Integrability
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            MEGPrep is designed with modularity in mind, allowing users to customize their preprocessing workflows easily. It integrates seamlessly with various libraries, including mne-python, enhancing its functionality for processing and analyzing MEG data.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Acceleration and Parallelization
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            By using the Nextflow framework, MEGPrep dramatically accelerates every step of the preprocessing pipeline. It is optimized for high parallelization, capable of managing heavy workloads and significantly speeding up data processing through concurrent execution of tasks. This capability is essential for conducting large benchmarks and effective comparisons across various tools and methodologies.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Interoperability and standards
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            MEGPrep includes an interactive reporting feature based on Streamlit, allowing users to visualize quality control metrics at each processing step. These reports provide alerts for any anomalies detected in the data, ensuring that the quality of each stage of processing is maintained.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Parameter Configuration
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            MEGPrep offers an easy-to-use configuration system, allowing users to specify parameters simply and intuitively. This configurability empowers researchers to adapt the preprocessing pipeline to their unique datasets and experimental needs without complex coding.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Automated Processes
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            MEGPrep enhances automated detection processes, including Automatic Artifacts Rejection, ICA (Independent Component Analysis) Automatic Detection, and auto-coregistration. These automated features streamline the preprocessing steps, reduce manual intervention, and improve accuracy and efficiency in artifact handling and data integration.

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/installation.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`bolt;2em` Guide for Beginner
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/quick_guide.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Concepts
         :class-card: sd-text-black sd-bg-light
         :link: details/pipeline_details.html



..   .. grid-item::
      :columns: 6 6 6 4

..      .. card:: :material-regular:`settings;2em` Quality Reference
         :class-card: sd-text-black sd-bg-light
         :link: quality_reference.html


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Quickstart

   quickstart/installation
   quickstart/quick_guide.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorial/tutorial_local.rst
   tutorial/tutorial_cluster.rst

   tutorial/reports.rst
   tutorial/outputs.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Pipeline Details

   details/pipeline_details.rst




