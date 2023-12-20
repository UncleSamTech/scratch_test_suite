# Nope Scratch That: A Historical Datasets of Revisions to Scratch Files
>> This is the code base needed to replicate the DataPaper on `Nope Scratch That: A Historical Datasets of Revisions to Scratch Files. This is built off this project [Opening the Valve on Pure Data: Collecting the data, creating the csvs, populating the database, and the usage of the database](https://github.com/anishaislam8/visual_code_revisions) It contains the following elements 
- Information Retreival Scripts from World of Code

- Commit Details Information Retreival such as:

    - Commit Messages
    - Author Information
    - Commit Information
    - Content Parents

- Database Construction.

>> Install all the dependencies needed for the project by running `pip3 install requirements.txt`

## Methodology of our Project. 
![Screenshot](/files/msr_flow-1.png)

## Usage of the Database(scratch_revision_db)

### Schema of our Database
![Screenshot](/files/Schema%20(1)-1.png)
### 1. Activate the sqlite command line interface
Our database is embedded in sqlite giving you the flexibility to query our datasets on the go without standing up a seperate server. 
>> Load the database using the command : `sqlite3 scratch_revisions_database.db`. This will automatically activate the command line for curating several queries

### 2. SAMPLE QUERIES

>> Example 1: Load five projects with the highest commit count
```
SELECT Project_Name, Total_Commits
FROM Projects
ORDER BY Total_Commits DESC
LIMIT 5;
```

>> Example 2: Get all information about a Scratch file `aa desafio1.sb3` relating to project to `logica-de-programacao-em-scratch`
```
SELECT * 
FROM REVISIONS
WHERE Project_Name = "logica-de-programacao-em-scratch" and File = "aa desafio1.sb3";
```

>> Example 3: Get the parsed content of the revision of a Scratch file `aa desafio1.sb3` from the Contents table where the project name is `logica-de-programacao-em-scratch` and commitid is `e6c8652392eff332a8176aeff1ccda2da7006d8a`

```
SELECT cont.Content
FROM Contents cont
JOIN Revisions rev ON cont.Hash = rev.Hash
WHERE rev.Project_Name = "logica-de-programacao-em-scratch" and rev.File = "aa desafio1.sb3" and rev.Commit_SHA = "e6c8652392eff332a8176aeff1ccda2da7006d8a";
```

>> Example 4 : Get all authors of the project `igorkowalczyk.github.io` that modifed the Scratch file

```
SELECT (distinct aut.Author_Name)
FROM Authors aut
JOIN Revisions rev ON aut.Commit_SHA = rev.Commit_SHA
WHERE rev.Project_Name = "igorkowalczyk.github.io";
```

>> Example 5: Get all commit messages of the unique commit id of the project `igorkowalczyk.github.io`

```
SELECT DISTINCT(com.Commit_SHA), com.Commit_Message
FROM Commit_Messages com
JOIN Revisions rev ON com.Commit_SHA = rev.Commit_SHA
WHERE rev.Project_Name = "igorkowalczyk.github.io";
```

>> Example 6: Get three projects with the highest number of Scratch files and show their total commit count

```
SELECT 
    proj.Project_Name,
    proj.Total_Commits
FROM
    Projects proj
JOIN (
    SELECT
        Project_Name,
        COUNT(DISTINCT File) as File_Count
    FROM
        Revisions
    GROUP BY
        Project_Name
    ORDER BY
        File_Count DESC
    LIMIT 3
) rev ON proj.Project_Name = rev.Project_Name;
```

### 3. Get the JSON Content of the revision of a Scratch file
>> Use the query below to view the content of scratch file `aa desafio1.sb3` located in project `logica-de-programacao-em-scratch` and has commit id `e6c8652392eff332a8176aeff1ccda2da7006d8a`


```
SELECT cont.Content
FROM Contents cont
JOIN Revisions rev ON cont.Hash = rev.Hash
WHERE rev.Project_Name = "logica-de-programacao-em-scratch" and rev.File = "aa desafio1.sb3" and rev.Commit_SHA = "e6c8652392eff332a8176aeff1ccda2da7006d8a";
```

### 4. Unzip the git repos of the scratch project (scratch_mirrored.tar.gz)

This is a large file with a size of x gig. Check for enough memory on your computer before unzipping it

>> Change to destination folder : Go to your target location where git repos would be unzipped and run the command on the command line `cd <destination folder>`

>> Unzip the Repos :  Run this command to unzip `tar -xzf scratch_mirrored.tar.gz`

### 5. Get the raw contents of a Scratch file
Getting the raw contents of a scratch file named `aa desafio1.sb3` from project `logica-de-programacao-em-scratch` with commit id `e6c8652392eff332a8176aeff1ccda2da7006d8a` involves the following steps

>> Change to the project directory : `cd logica-de-programacao-em-scratch`

>> Show the raw contents of the Scratch file `git show e6c8652392eff332a8176aeff1ccda2da7006d8a:"aa desafio1.sb3" `

### 6 Manually parse a scratch file
You can parse a Scratch file by looking into the scratch_parser.py script

>> Parse a Scratch file using this command `python3 scratch_parser.py <file_name>`. This generates an Abstract Syntax tree saved in `/scratch_tester/scratch_test_suite/files` folder as a json file with the name of the Scratch file used. For instance if pass a Scratch file `scratch_simulate.sb3`, it would generate a `scratch_simulate.json` file in that folder.



