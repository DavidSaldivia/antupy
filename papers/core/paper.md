---
title: "Antupy: A Python package for energy engineering simulations"
authors:
  - name: David Saldivia
    affiliation: 1
affiliations:
  - name: Solar Energy Research Center (SERC) Chile.
    index: 1
date: 13 November 2025
bibliography: antupy_references.bib
---

# Summary

`antupy`, from Mapuche language *antü*, meaning "sun"[@wiki_mapuche], is a Python package designed as a toolkit for energy system simulations. The package provides a framework organised around three type of classes. The core data types (`Var`, `Array`, `Frame`) for handling physical quantities with automatic unit conversion; abstract protocol classes (`Model`, `Plant`, `Analyser`, `TimeSeriesGenerator`) that enable modular and extensible simulation workflows; and a suite of utility modules for thermophysical properties (`props`), heat transfer correlations (`htc`), and solar calculations (`solar`). Built on top of established scientific libraries including NumPy [@numpy], polars [@polars], and SciPy [@scipy]. This paper focuses on the core unit management system, some of the utility modules, and the abstract protocol architecture that enables researchers to develop custom energy system models with minimal boilerplate code while maintaining dimensional consistency throughout their simulations and post-processing.

# Statement of need

This software targets energy and mechanical engineering education and research. From undergraduates taking their first basic science courses to active researchers, computational tools that balance accessibility, flexibility, and rigor are essential for solving engineering problems. Energy and mechanical engineering programs increasingly rely on computational methods to teach thermodynamic cycles, heat transfer, and renewable energy systems. Simultaneously, researchers working on solar energy deployment need flexible frameworks to prototype novel system configurations, conduct parametric studies, and validate experimental results. These two domains—education and research—share a common need for tools that are both pedagogically transparent and sufficiently powerful for real-world applications.

Established energy simulation platforms such as System Advisor Model (SAM) [@nrel_sam], Engineering Equation Solver (EES) [@fchart_ees], and TRNSYS [@trnsys] have proven invaluable for industry applications and detailed system modeling. However, these tools present barriers for Python-based workflows, which have become the de facto standard in data science, machine learning, and modern scientific computing. While Python packages like TESPy [@witte2020tespy] provide thermal system modeling capabilities and pvlib [@holmgren2018pvlib] excels at photovoltaic performance simulation, there remains a gap for a general-purpose framework specialized in annual energy simulations that enables researchers to implement custom modules, control solvers, and integrate diverse energy technologies within a unified architecture. Existing Python tools often focus on specific technologies or require significant overhead to extend beyond their original scope.

A fundamental challenge in energy system modeling is the management of physical units across calculations involving thermodynamics, heat transfer, and fluid mechanics. Engineers routinely work with temperatures in Celsius and Kelvin, pressures in Pascals and bar, heat transfer rates in Watts and kW, and must ensure dimensional consistency when combining thermophysical properties from sources like CoolProp, convection correlations, and solar radiation models. While Python packages for unit management exist—such as Astropy [@astropy], Pint [@pint], and forallpeople [@forallpeople]—these tools do not seamlessly integrate unit variables across scalar (`Var`), vector (`Array`), and tabular (`Frame`) data structures. Furthermore, `antupy` employs simple, standard unit labels following intuitive rules (detailed in the documentation) that reduce cognitive overhead for engineers. By combining thermophysical property evaluation, heat transfer coefficient calculations, and solar geometry routines with automatic unit tracking and conversion, `antupy` provides a cohesive framework where physical quantities carry their units throughout the simulation workflow, reducing errors and improving code readability while integrating essential utilities for energy engineering into a single, coherent package.

# Implementation

The `antupy` package architecture (Figure 1) is organized around core data structures with two derived functional groups. The **core layer** provides three immutable data types: `Var` for scalar physical quantities, `Array` for homogeneous vector data (leveraging NumPy arrays), and `Frame` (extending polars DataFrame) for tabular data with per-column unit tracking. These types support arithmetic operations with automatic unit conversion and dimensional checking—for example, adding `Var(5.0, "kg")` and `Var(500, "g")` correctly yields `5.5 [kg]`. Building on these core classes, the **utilities modules** provide domain-specific functionality: `props` (thermophysical properties via CoolProp and Cantera), `htc` (heat transfer correlations for natural and forced convection), `solar` (sun position and radiation calculations), and `loc` (geographical location management with Australian and Chilean databases). The **protocol/abstract classes** define contracts for extensibility: `Model` (component-level solvers), `TimeSeriesGenerator` (weather and market data), `Plant` (system integration), and `Analyser` (parametric studies). Both the utilities and protocols are designed to be fully compatible with the core unit-aware data structures. Current implementations include domestic water heating (`dwh`) and concentrated solar power (`cst`) modules that demonstrate the framework's application to real-world systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ANTUPY PACKAGE                          │
├─────────────────────────────────────────────────────────────────┤
│  CORE LAYER: Unit Management                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐     │
│  │   Var    │  │  Array   │  │  Frame (extends pandas)  │     │
│  │ (scalar) │  │ (vector) │  │  (tabular w/ units)      │     │
│  └──────────┘  └──────────┘  └──────────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  UTILITIES LAYER                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  props   │  │   htc    │  │  solar   │  │   loc    │      │
│  │(CoolProp)│  │(conv/rad)│  │(geometry)│  │ (geo DB) │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│  SIMULATION LAYER: Protocol Classes                             │
│  ┌────────────────┐  ┌──────────────────────────────┐         │
│  │ Model / Plant  │  │ TimeSeriesGenerator (TSG)   │         │
│  │ (components)   │  │ (weather, loads, markets)   │         │
│  └────────────────┘  └──────────────────────────────┘         │
│  ┌───────────────────────────────────────────────────┐         │
│  │ Analyser: Parametric (sensitivity analysis)       │         │
│  │           MonteCarlo (uncertainty propagation)    │         │
│  └───────────────────────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│  IMPLEMENTATIONS: dwh (domestic water heating)                  │
│                   cst (concentrated solar thermal)              │
└─────────────────────────────────────────────────────────────────┘
```
**Figure 1:** Architecture of the `antupy` package showing the three-layer design with core unit management, utility modules, and extensible protocol-based simulation framework.

The package is designed for minimal friction in typical workflows. A simple example calculating heat loss from a solar collector demonstrates the unit-aware approach:

```python
from antupy import Var
from antupy.utils.htc import h_horizontal_surface_upper_hot

temp_surface = Var(80, "°C")
temp_ambient = Var(25, "°C")
area = Var(2.5, "m2")

# Calculate convection coefficient (returns Var with W/m2-K)
h_conv = Var(h_horizontal_surface_upper_hot(
    temp_surface.gv("K"), temp_ambient.gv("K"), L=1.5
), "W/m2-K")

# Heat loss calculation with automatic unit handling
q_loss = h_conv * area * (temp_surface - temp_ambient)
print(f"Heat loss: {q_loss.gv('W'):.1f} W")  # Convert to Watts for display
```

The `Parametric` analyser enables systematic sensitivity studies by automatically managing parameter combinations and preserving units in results tables (stored as `Frame` objects). Current `TimeSeriesGenerator` implementations include market price data for Australia and Chile, as well as weather data through TMY (Typical Meteorological Year) and historical weather datasets. Comprehensive documentation is available online, including detailed introductions to the core variables, introductory examples, usage guides for the utility libraries, and the complete API reference. The package requires Python ≥3.12.

# Current and future development

The `antupy` codebase represents a formalization and generalization of simulation frameworks developed during concentrated solar power (CSP) and domestic water heating (DWH) research projects [@mcrt; @dwh]. These initial applications validated the core architecture and identified common patterns that informed the abstract protocol design. Current development priorities include migrating additional project-specific modules—such as desalination systems (`desal`) and compressed air energy storage (`caes`)—into the unified framework. Future enhancements will expand the `TimeSeriesGenerator` ecosystem to include additional weather data sources, electricity market price signals, and load profile generators for diverse building types. Solver implementations will grow to encompass more renewable energy technologies while maintaining the lightweight protocol-based coupling that enables researchers to integrate external solvers (including commercial tools like TRNSYS) through consistent interfaces. Community contributions are welcomed through the project's GitHub repository, with emphasis on maintaining the balance between educational clarity and research-grade capability that defines `antupy`'s design philosophy.

# Acknowledgments

The author expresses gratitude to the projects ANID/FONDAP/1523A0006 "Solar Energy Research Center"—SERC-Chile and ANID's scholarship program "Becas Chile" (grant number TBD). <!-- Suggest a change here--> Additionally, with the name, the author acknowledges the Mapuche people as an inspiration. The first beta version of this codebase was written in Temuco, located in the heart of the Araucanía region, traditional territory of the Mapuche people.

# References