"""Core application modules.

Layering, from the outside in:

* ``gui``                Qt interface (pages, theme, widgets)
* ``benchmark``          performance measurement and reporting
* ``pipeline``           realtime capture / inference / display architecture
* ``application``        service assembly shared by the GUI and the CLI
* domain                camera, detector, quality, guidance, registration,
                        recognition, similarity, database, liveness
* ``settings``/``performance``/``hardware``  configuration resolution
* ``metrics``/``sysmetrics``                 measurement primitives

Submodules are imported explicitly (``from src.pipeline import ...``) rather than
re-exported here, so importing one module never drags in the rest.
"""
