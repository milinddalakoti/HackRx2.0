import mysql.connector

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insertBLOB(emp_id, name, photo):
    print("Inserting BLOB into Python_Employee table")
    connection = mysql.connector.connect(host='localhost',
                                             database='db1_test',
                                             user='root',
                                             password='Password@123')
    try:
        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO Python_Employee
                          (id, name, photo) VALUES (%s,%s,%s)"""

        empPicture = convertToBinaryData(photo)
        
        # Convert data into tuple format
        insert_blob_tuple = (emp_id, name, empPicture)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into python_employee table", result)

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

insertBLOB(1, "First", "/home/pradyumn/scripts/hackathons/Hackrx/images/images.jpeg")


class retrieve():
    
    def write_file(self,data, filename):
        # Convert binary data to proper format and write it on Hard Disk
        with open(filename, 'wb') as file:
            file.write(data)


    def readBLOB(self,emp_id, photo, bioData):
        print("Reading BLOB data from python_employee table")

        try:
            connection = mysql.connector.connect(host='localhost',
                                                 database='db1_test',
                                                 user='root',
                                                 password='Password@123')

            cursor = connection.cursor()
            sql_fetch_blob_query = """SELECT * from python_employee where id = %s"""

            cursor.execute(sql_fetch_blob_query, (emp_id,))
            record = cursor.fetchall()
            for row in record:
                print("Id = ", row[0], )
                print("Name = ", row[1])
                image = row[2]
                file = row[3]
                print("Storing employee image and bio-data on disk \n")
                write_file(image, photo)
                write_file(file, bioData)

        except mysql.connector.Error as error:
            print("Failed to read BLOB data from MySQL table {}".format(error))

        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

'''
readBLOB(1, "D:\Python\Articles\my_SQL\query_output\eric_photo.png",
             "D:\Python\Articles\my_SQL\query_output\eric_bioData.txt")
readBLOB(2, "D:\Python\Articles\my_SQL\query_output\scott_photo.png",
             "D:\Python\Articles\my_SQL\query_output\scott_bioData.txt") 
'''