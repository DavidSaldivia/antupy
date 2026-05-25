
Mail Jose/Rodrigo
=========================

Hola José/Rodrigo,

Espero este email los encuentre bien. Se los envío a partir de lo que he conversado con ustedes por separado sobre antupy. Quiero trabajar en un paper sobre la librería y quiero invitarlos como coautores. El target journal aun no lo tengo claro, pero puede ser alguno en educación en ingeniería (hay que buscar uno) o algo como energy conversion and management, yo creo que depende del enfoque que le queramos dar.
Mi idea para la estructura del paper es presentar primero el software en general, con sus tres capas (core, utils, simulation, que estaría a cargo mío), y luego un conjunto de casos de estudio simples en distintos campos de la ingeniería energética. Esto último constituirá el "catalog" del software (la cuarta capa, a cargo de los otros coautores). Para cada sección del catálogo me gustaría tener un encargado que sea un experto del área (ver lista tentativa abajo) y que serán los coautores del paper.
Como métodología de trabajo, propongo tener reuniones uno-a-uno con los otros autores donde trabajemos un problema en particular. El problema debe ser relativamente simple y clásico de cierto campo de estudio, estilo/dificultad de curso de pregrado que enseñen. La idea es implementarlo en python usando antupy. La idea es tener estos 3 o 4 ejemplos luego de unos 4 meses.

Coautores propuestos, según contribución a catalog:
- cycles: David Saldivia (modulo con componentes basicos de ciclos termodinámicos)
- csp: Robert A. Taylor (example based on my phd code)
- dwh: Anna Bruce, Baran Yildiz (based on tm_solarshift)
- stg: José Cardemil (crear un wrapper de algún modelo de SHIPcal?)
- cryo: Rodrigo Barraza (ya conversamos sobre algún problema de refrigeración.)
- desal: Amr Omar (con él trabajé durante mi phd,  tradujo mi modelo MED a matlab)
- chem: Felipe Huerta (no he hablado directamente con él, pero creo que le interesaría)

Obviamente, si ustedes consideran algún otro coautor que quiera trabajar directamente en algún modulo, yo feliz.

Para iniciar esto formalmente, lo ideal sería tener una reunión en conjunto (o quizás dos, una en horario AU y otro en CL). Yo estaría pensando para la segunda o tercera semana de junio, dependiendo de la disponibilidad. En esa reunión yo haría un tutorial corto de la librería y propondría un plan de trabajo para ser discutido y continuar desde ahí.


Mail Robert/Anna/Bruce
=========================

Co-authorship paper proposal

Hi XXXXXX,

I hope this email finds you all well. I bit of time without getting in touch. First, I'd like to update you from my side. I have started in a new position as a postdoc in CENTRA in UAI. I'll be working on a couple of projects, but the main one is a 3-years project to assess the potential and propose a roadmap for VPPs in Chile. I'm also lecturing two courses in UDP: Thermodynamics and Energy Project Assessment. So, I'll be around in academia for a couple of years more at least.

I'm reaching out to invite you for a paper coauthorship. Since last year, I've been working in a Python library to support (energy) engineering simulations. The library is called antupy, and it's already published in PyPI (which means you can install it with "pip install antupy"). You can see the open repository [here](https://github.com/DavidSaldivia/antupy), while the official documentation is [here](https://antupy.readthedocs.io). Well, the origins of this library are from my PhD code, when I was just dropping commonly used solar functions in a single file, and it was during my work in solarshift that I started to think on what I could finally formalise during last year. So, naturally, I think it would be great if you can be part of this paper.

For the paper structure, my idea is to split it in two parts: the general explanation of the software (antupy itself) and a presentation of several use cases (what I call the catalog). I will be in charge of writing the first part, and the idea is to have each coauthor, as an expert in their field, in charge of one section of the catalog. Each use case should be an example of simulation of a simple engineering system using antupy. In your cases, it should be simpler, because I have already updated parts of my phd code and tm_solarshift, using this new library. Based on your availability, we can plan a work plan(?), but the minimal contribution I would expect is to write that subsection.

The modules I'm expecting to include in the catalog, with a proposed expert in charge, and a short comment:
- cycles (D. Saldivia), some basic models of typical components (turbines, pumps, compressors) used in thermodynamic cycles. 
- csp: Robert A. Taylor, some example of a csp plant, based on my phd code.
- dwh: Anna Bruce, Baran Yildiz, some example based on tm_solarshift.
- stg: José Cardemil, he is the 
- cryo: Rodrigo Barraza (ya conversamos sobre algún problema de refrigeración.)
- desal: Amr Omar (con él trabajé durante mi phd,  tradujo mi modelo MED a matlab)
- chem: Felipe Huerta (no he hablado directamente con él, pero creo que le interesaría)

Please, let me know what do you think


Roughly speaking, these four layers are structured as follow:
- core layer: it is the basic classes: Var, Array and Frame. They are the heart of the library and is used by all the resto of the ecosystem. I'd expect all the authors to have a basic understanding of this.
- utils layer: It is a set of modules with utilities such as props (thermophysical properties, a CoolProp wraper), htc (heat transfer correlations), solar (solar position, incidence angle calculator, radiation models, etc.).
- simulation layer: Here are some classes to help simulate real system, for example, Plant (to integrate components) or Parametric (to run parametric analysis over Plant's simulations).
- catalog: It's where the models are stored. My plan is to have several folders, each with example models of different engineering systems. Each folder has someone in charge, that's an expert in the area. Some example of possible folders (with a proposed person in charge), are:
    - cycles: David Saldivia. I'm developing this one as a example, using typical thermodynamic cycles, such as Rankine, Brayton.
    - csp: Robert A. Taylor. This can be based on my phd.
    - dsw: B. Yildiz / A. Bruce. This can be based on tm_solarshift.
    - desal: Amr Omar. This can be based on my master code.
    - cryo: R. Barraza.
    - stg: J. Cardemil.
    - chem: Not defined (Rob Patterson, Felipe Huerta)

My proposed workflow would be something like this:
- A kick-off meeting with all/most coauthors with the following topics:
    - 

The target journal: Still not defined, something for engineering education or a general energy engineering paper (ECM, Renewable Energy, ). It depends on how novel we think this is for research or if it is more suitable for education.