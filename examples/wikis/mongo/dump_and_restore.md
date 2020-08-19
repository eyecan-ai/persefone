# Mongo Database

## Dump

dump a database with `$DATABASE_NAME` into a single archive file:

```
mongodump --db=$DATABASE_NAME --archive=$ARCHIVENAME
```

## Restore

```
mongorestore --archive=$ARCHIVENAME --nsFrom "$DATABASE_NAME.*" --nsTo "$NEW_DATABASE_NAME.*"

```