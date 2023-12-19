# Nope Scratch That: A Historical Datasets of Revisions to Scratch Files
>> This is the code base needed to replicate the DataPaper on `Nope Scratch That: A Historical Datasets of Revisions to Scratch Files. This is built off of on this project [Opening the Valve on Pure Data: Collecting the data, creating the csvs, populating the database, and the usage of the database](https://github.com/anishaislam8/visual_code_revisions) It contains the following elements 
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
- Load the database using the command : `sqlite3 scratch_revisions_database.db`. This will automatically activate the command line for curating several queries

### 2. SAMPLE QUERIES

>> Example 1: Load five projects with the highest commit count
```
SELECT Project_Name, Total_Commits
FROM Projects
ORDER BY Total_Commits DESC
LIMIT 5;
```

>> Example 2: Get all information about a Scratch file `lib/szkola/2019-2020/praca/Obliczenia.sb3` relating to project to `igorkowalczyk.github.io`
```
SELECT * 
FROM REVISIONS
WHERE Project_Name = "igorkowalczyk.github.io" and File = "lib/szkola/2019-2020/praca/Obliczenia.sb3";
```

>> Example 3: Get the parsed content of the revision of a Scratch file `lib/szkola/2019-2020/praca/Obliczenia.sb3` from the Contents table where the project name is `igorkowalczyk.github.io` and commitid is `013a7ab6887e32ceca2066fc544726c6e5499e463ed94c68e08`

```
SELECT cont.Content
FROM Contents cont
JOIN Revisions rev ON cont.Hash = rev.Hash
WHERE rev.Project_Name = "igorkowalczyk.github.io" and rev.File = "lib/szkola/2019-2020/praca/Obliczenia.sb3" and rev.Commit_SHA = "013a7ab6887e32ceca2066fc544726c6e5499e463ed94c68e08";
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