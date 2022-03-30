# ICS - Interactive Classification System

## Install

Installation steps:
* install code (using pip or from source)
* db configuration

### Install code: using pip

ICS is published as a [`pip` package](https://pypi.org/project/icspkg)

```
> pip install icspkg
```

### Install code: from source
Download source code from [github repo](https://github.com/aesuli/ics).

The suggested way to quickly setup the python enviroment is to use the Anaconda/Miniconda distribution and the `conda` package manager to create the virtual enviroment.

```
> cd [directory with ICS code]
> conda create -n ics --file requirements.txt
```

### DB configuration

This step is required either if you installed ICS using pip or from source.
ICS requires a database to store its data.

ICS is tested to work with [PostgreSQL](https://www.postgresql.org/).
The `db.psql` file contains the SQL commands to create the required database on PostgreSQL.

```
    CREATE USER ics WITH PASSWORD 'ics';
    CREATE DATABASE ics;
    GRANT ALL PRIVILEGES ON DATABASE ics to ics;
```

```
> psql -f db.psql
```

The db data structures required by the applications are created, if missing, at the first run.

## Starting

_Activate the virtual environment_
```
> conda activate ics
```
If installed using pip, the main application can be started with the command:
```
> ics-webapp
```
From the root directory of a source code copy, it can be launched from the `ics-webapp.py` script:
```
Linux/Mac:
>PYTHONPATH=. python ics/scripts/ics-webapp.py

Windows:
>set PYTHONPATH=. 
>python ics/scripts/ics-webapp.py
```
When launched, the app will print the URL at which it is accessable.
```
[30/Mar/2022:15:31:59] ENGINE Bus STARTING
[30/Mar/2022:15:31:59] ENGINE Started monitor thread 'Autoreloader'.
[30/Mar/2022:15:31:59] ENGINE Serving on http://127.0.0.1:8080
[30/Mar/2022:15:31:59] ENGINE Bus STARTED
[30/Mar/2022:15:31:59] ENGINE Started monitor thread 'Session cleanup'.
```
After the installation, only the `admin` user is defined, with password `adminadmin`.

## Configuration

A configuration for `ics-start` can be saved to a file using the `-s` argument with the filename to use.
For example, this command creates a `default.conf` file that lists all the default values (if any other argument is used in the command, the value of the argument is saved in the configuration file).

```
> ics-start -s default.conf
```


A configuration file can be used to set the launch arguments, using the `-c` argument:

```
> ics-start -c myinstance.conf
```

Any additional argument passed on the command line overrides the one specified in the configuration file.

## License

This software is licensed under the [3-Clause BSD license](https://opensource.org/licenses/BSD-3-Clause) unless otherwise noted.