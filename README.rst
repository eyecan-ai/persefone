=========
persefone
=========






Python library for deep learning data manipulation



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage



MONGO
-----

dump:
```
mongodump --db=DATABASE_NAME --archive=ARCHIVENAME
```

restore:
```
mongorestore --archive=<ARCHIVENAME> --nsFrom "OLD_DATABASE_NAME.*" --nsTo "NEW_DATABASE_NAME.*"

```