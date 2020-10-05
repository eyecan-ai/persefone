

MONGO
-----

dump:

.. code:: bash

    mongodump --db=DATABASE_NAME --archive=ARCHIVENAME

restore:

.. code:: bash

    mongorestore --archive=<ARCHIVENAME> --nsFrom "OLD_DATABASE_NAME.*" --nsTo "NEW_DATABASE_NAME.*"
