
Mail Jose/Rodrigo
=========================

Hola José  y Rodrigo,

Espero este email los encuentre bien. Se los envío a partir de lo que he conversado con ustedes por separado sobre antupy. Quiero trabajar en un paper sobre la librería y quiero invitarlos como coautores. El target journal aun no lo tengo claro, pero puede ser alguno en educación en ingeniería (hay que buscar uno) o algo como energy conversion and management, yo creo que depende del enfoque que le queramos dar.

Estructura del paper:
La idea es presentar primero el software en general, con sus tres capas (core, utils, simulation, que estaría a cargo mío), y luego un conjunto de casos de estudio simples en distintos campos de la ingeniería. Esto último constituirá el "catalog" del software (la cuarta capa, a cargo de los otros coautores). Para cada sección del catálogo me gustaría tener un encargado que sea un experto del área (ver lista tentativa abajo) y que serán los coautores del paper.

Metodología de trabajo:
Como métodología de trabajo, propongo tener reuniones uno-a-uno con los otros autores donde trabajemos un problema en particular. El problema debe ser relativamente simple y clásico de cierto campo de estudio, algo con una dificultad similar a la de algún curso de pregrado que enseñen. La idea es implementarlo en python usando antupy y tener estos 3 o 4 ejemplos luego de unos 4 meses. En esas sesiones definiríamos el problema, veremos como implementarlo y definir qué resultados mostrar. Yo apoyaría en la implementación, mientras que la redacción final de cada subsección estaría a cargo de los coautores. Mi objetivo es tener una versión final del paper de acá a fin de año.

Coautores invitados:
Coautores propuestos, según contribución a catalog:
- thermo: David Saldivia (modulo con componentes basicos de ciclos termodinámicos)
- csp: Robert A. Taylor (ejemplo basado en el código de mi phd)
- dwh: Anna Bruce, Baran Yildiz (ejemplo basado en tm_solarshift)
- stg: José Cardemil (crear un wrapper de algún modelo de SHIPcal?)
- cold: Rodrigo Barraza (ya conversamos sobre algún problema de refrigeración.)
- desal: Falta definir encargado. Por ahora sería yo y trataría de adaptar el modelo MED que desarrollé en mi master. Con Rodrigo hemos desarrollado unos modelos de HDH en EES, que podrían ser migrados.
- chem: Felipe Huerta (no he hablado directamente con él, pero creo que le interesaría)

Obviamente, si ustedes consideran algún otro coautor que quiera trabajar directamente en algún modulo, yo feliz de incorporarlo. Con respecto a los coautores de australia (Rob, Anna y Baran), pretendo enviarles un email similar a este para invitarlos.

Para iniciar esto formalmente, lo ideal sería tener una reunión en conjunto (o quizás dos, una en horario AU y otro en CL). Yo estaría pensando para fines de junio o principio de julio, dependiendo de la disponibilidad. En esa reunión yo haría un tutorial corto de la librería y propondría un plan de trabajo para ser discutido y continuar desde ahí.


Mail Robert/Anna/Bruce
=========================

Co-authorship paper proposal

Hi XXXXXX,

I hope this email finds you all well. I bit of time without getting in touch. First, I'd like to update you from my side. I have started in a new position as a postdoc in the Center for Energy Transition (CENTRA) at the Universidad Adolfo Ibañez (UAI), in Santiago. I'll be working on a couple of projects, but the main one is a 3-years project to assess the potential and propose a roadmap for VPPs in Chile. I'm also lecturing a couple of courses, so I'll be around in academia for a couple of years more at least.

I'm reaching out to invite you for a paper coauthorship. Since last year, I've been working in a Python library to support (energy) engineering simulations. The library is called antupy, and it's already published in PyPI (which means you can install it with "pip install antupy"). You can see the open repository [here](https://github.com/DavidSaldivia/antupy), while the official documentation is [here](https://antupy.readthedocs.io). Well, the origins of this library are from my PhD code, when I was just dropping commonly used solar functions in a simple module, and it was during my work in solarshift that I started to formulate what I could finally formalise last year. So, naturally, I think it would be great if you can be part of this publication.

For the paper structure, my idea is to split it in two parts: the general explanation of the software (antupy itself) and a presentation of several application examples (what I call the catalog). I will be in charge of writing the first part, and the idea is to have each coauthor, as an expert in their field, in charge of one section of the catalog. Each use case should be an example of a simple engineering simulation using antupy. In your cases, it should be simpler, because I have already updated parts of my phd code and tm_solarshift, using this new library. Based on your availability, we can define a working plan, but the minimal contribution I would expect from your side is to write that subsection.

The modules I'm expecting to include in the catalog, with a proposed expert in charge, and a short comment:
- thermo: D. Saldivia. Some basic models of typical components (turbines, pumps, compressors) used in thermodynamic cycles. 
- csp: Robert A. Taylor, some example of a csp plant, based on my phd code.
- dwh: Anna Bruce, Baran Yildiz, some example based on tm_solarshift.
- stg: José Cardemil. He is the chief director of SERC Chile, and an expert in thermal storage and solar energy. I worked with him last year
- cold: Rodrigo Barraza. He is the chief director of CENTRA 
Another potential modules are desal, chemical, etc. if you think there are other researchers that could be interested on this project, please, let me know.


My proposed workflow would be to have a kick-off meeting with all/most coauthors. In this meeting I'd explain the software architecture and philosophy, and propose a working plan. Then, we organize a couple of 1v1 meetings with the coauthors to work on their specific module. My idea is to have 3-4 examples in around 4 months, and have a "submittable" version by the end of the year.

I haven´t defined a target journal yet, but could be one for engineering education or a general energy engineering paper (ECM, Renewable Energy, etc). It depends on how novel we think this is for research or if it is more suitable for education.