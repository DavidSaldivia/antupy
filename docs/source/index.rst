.. Antupy documentation master file, created by
   sphinx-quickstart on Wed Jul 23 22:48:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Antupy documentation
====================

antupy (pronounced *antu-paɪ* [1]_ , from the mapudungún word *"antü"* (sun) [2]_ ) is an open-source python library to analyse (thermal) energy systems.

It includes a series of classes and methods to simulate energy conversion and energy storage systems, under uncertain timeseries constraints (weather, market, human behaviour, etc.).

It is an object-oriented software, with a unit manager in its core, creating two classes: ``Var`` and ``Array``, to represent escalars and vectors (or timeseries), respectively. To help simulate real world systems, it provides four main classes: ``Models``, ``Plants``, ``Timeseries Generators``, and ``Analysers``. The different analysers allow a wide range of outputs such as: technical, economics, financial, environmental (emissions), etc. It also include a toolbox with classes and functions like an unit conversion system, a thermophysical properties library, and a heat transfer coefficient library.


.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   units
   variable_system
   props
   api



.. [1] IPA pronunciation.

.. [2] mapudungún is the language of the Mapuche people, the main indigineous group in Chile. *antü* means sun, but it also represents one of the main *pilláns* (spirits) in the Mapuche mythology. Here the word is used with its first literal meaning. The name was chosen because the first version of this library was written in Temuco, a Chilean city located at Mapuche heartland (*Wallmapu*).