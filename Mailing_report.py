import smtplib, ssl, email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from email.mime.application import MIMEApplication
from datetime import datetime
import socket



def Send_email():
    sender_email = "trainervadim1@gmail.com"
    receiver_email = "vadim.gliner@gmail.com"
    message = 'Subject: {}\n\n{}'.format('Training complete - RBBB', 'Training is complete, congradulations')
    port = 465  # For SSL
    password = 'Vg25343021'

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("trainervadim1@gmail.com", password)
        server.sendmail(sender_email, receiver_email, message)

def summarize_learning_file(path_):
    data = np.genfromtxt(path_, delimiter = '\t')
    data = np.array(data[1:][:])
    Summary = ''
    #pandas
    df=pd.read_csv(path_, sep='\t', dtype={'IS_BEST_SRC_ACC': bool})  #,header=None  
    is_checkpointed = df['IS_BEST_SRC_ACC']

    if len(is_checkpointed) < 1:
        Summary = Summary +'Not enough epochs for learning\n LEARNING FAILED'
        return (Summary,0)
    if np.sum(is_checkpointed) == 0 :
        Summary = Summary +'Tried to learn but no checkpoints\n LEARNING FAILED'
        return (Summary,0)    


    b = is_checkpointed[::-1]
    i = len(b) - np.argmax(b) - 1
    fig1 = plt.figure()
    plt.plot(data[:,0],data[:,5])
    plt.plot(data[:,0],data[:,6])
    plt.scatter(i,data[i,6],alpha=0.5,marker="X",c="r")
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.title('Accuracy vs. Epoch- Classifier')
    plt.legend(['Training', 'Test'])
    plt.savefig(os.path.join(os.getcwd(),"Training_curve.jpg"), dpi=300)
    plt.clf()
    plt.plot(data[:,0],data[:,9])
    plt.plot(data[:,0],data[:,10])
    plt.scatter(i,data[i,10],alpha=0.5,marker="X",c="r")
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.title('Accuracy vs. Epoch- Discriminator')
    plt.legend(['Training', 'Test'])
    plt.savefig(os.path.join(os.getcwd(),"Training_curve_disc.jpg"), dpi=300)

    print(f'Index of last checkpoint: {i}\n')
    Summary= Summary+f'Total number of epochs : {len(data[:,0])}\n\n'
    Summary= Summary+f'Epoch of last checkpoint: {i}\n\n'
    Summary= Summary+f'Training:  Started from: {data[0,5]}, reached {data[i,5]}\n\n'
    Summary= Summary+f'Test:  Started from: {data[0,6]}, reached {data[i,6]}\n\n'
    Summary= Summary+f'Discriminator accuracy: {data[i,10]}\n\n'
    Summary= Summary+f'Server name: {socket.gethostname()}\n\n'
    return (Summary,1)

def Send_detailed_email(Disease='Normal ECG'):
    with open(os.path.join(os.path.dirname(__file__),f"Mailing_log_{Disease}.txt"), "a") as mailing_logger:
        now = datetime.now()
        mailing_logger.write(f'{now}:: Entering email routine\r\n')    
        subject = f"Training complete - {Disease}"
        body = f"Results summary:\n\n" + f"Server : {socket.gethostname()}\n\n"
        sender_email = "trainervadim1@gmail.com"
        receiver_email = "vadim.gliner@gmail.com"
        password = 'Vg25343021'
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        # os.path.dirname(__file__)
        suffix_ = Disease.replace(' ','_')
        mailing_logger.write(f'{now}:: Looking for raw data\r\n')    
        path_=os.path.join(os.getcwd(),'Logs',f'Execution_dump_kernel*{suffix_}*.txt')
        filenames= glob.glob(path_)
        is_success=0
        now = datetime.now()
        if len(filenames)> 0:
            mailing_logger.write(f'{now}:: Found raw data\r\n')    
        else:
            mailing_logger.write(f'{now}:: Didnt find raw data\r\n')    

        # Open file in binary mode
        if len(filenames)>1:
            filename = filenames[-1]
        else:
            filename = filenames
        data,is_success = summarize_learning_file(filename)
        body = body+ data
        message.attach(MIMEText(body, "plain"))
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())           
        # Encode file in ASCII characters to send by email    
            encoders.encode_base64(part)

        mailing_logger.write(f'{now}:: Finish processing raw data\r\n')    

    # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        if is_success:
            attachment = MIMEApplication(open(os.path.join(os.getcwd(),"Training_curve.jpg"), "rb").read())
            attachment.add_header('Content-Disposition','attachment', filename=os.path.join(os.getcwd(),"Training_curve.jpg"))
            message.attach(attachment)
            attachment = MIMEApplication(open(os.path.join(os.getcwd(),"Training_curve_disc.jpg"), "rb").read())
            attachment.add_header('Content-Disposition','attachment', filename=os.path.join(os.getcwd(),"Training_curve_disc.jpg"))
            message.attach(attachment)

        text = message.as_string()    

        # Add body to email
        message.attach(MIMEText(body, "plain"))
        mailing_logger.write(f'{now}:: Mail construction finished\r\n')    
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)    
        
        mailing_logger.write(f'{now}:: Mail sent\r\n')    

if __name__ == "__main__":
    print('Try mailing...')
    Send_detailed_email('Sinus bradycardia')    
    # Send_email()
    print('End mailing...')
