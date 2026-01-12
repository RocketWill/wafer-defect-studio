# Keep project writes in the GUI service and run compute in worker processes

Project metadata will live in SQLite and only the GUI-side project service will write it. Training and detection run in separate worker processes that emit progress and stage artifacts for transactional registration, isolating the UI from GPU failures while avoiding concurrent database ownership and partial run records.
