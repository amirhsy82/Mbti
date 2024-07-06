# Importing Packages
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import re
import requests
import matplotlib.pyplot as plt
import io
import base64


# Creating App 
app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mbti.db'
db = SQLAlchemy(app)

# Database Model
class Message(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255), nullable=False)
    translated_message = db.Column(db.String(), nullable=False)
    cleaned_message = db.Column(db.String(), nullable=False)
    mbti_type = db.Column(db.String(4), nullable=False)

# load the ml model
EI_model = pickle.load(open(r"C:\Users\USER\Desktop\mbti\Resources (weights-vectorizer)\E-I.pkl.txt", 'rb'))
NS_model = pickle.load(open(r"C:\Users\USER\Desktop\mbti\Resources (weights-vectorizer)\N-S.pkl.txt", 'rb'))
FT_model = pickle.load(open(r"C:\Users\USER\Desktop\mbti\Resources (weights-vectorizer)\F-T.pkl.txt", 'rb'))
JP_model = pickle.load(open(r"C:\Users\USER\Desktop\mbti\Resources (weights-vectorizer)\J-P.pkl.txt", 'rb'))
# load the vectorizer model
vectorizer = pickle.load(open(r"C:\Users\USER\Desktop\mbti\Resources (weights-vectorizer)\vectorizer.pkl.txt", 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message_content = request.form['message']

        # Translate message 
        url = "https://api.edenai.run/v2/translation/automatic_translation"
        payload = {
            "response_as_dict": True,
            "attributes_as_list": False,
            "show_original_response": False,
            "text": message_content, 
            "source_language": "fa",
            "target_language": "en",
            "providers": "phedone"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZmUzOWZiYmMtYjE1Ni00YmJlLWI5N2YtN2RiODc3Njg4OTA5IiwidHlwZSI6ImFwaV90b2tlbiJ9.aUzU0kVy2DlB9hPWJi2afVR-l3bffeTEfLq1Is6GlLo"
        }

        response = requests.post(url, json=payload, headers=headers)
        response = response.json()
        translated = response['phedone']['text']


        # Text Cleaninig
        replacements = [
            (r"(http.*?\s)", " "),
            (r"[^\w\s]", " "),
            (r"\_", " "),
            (r"\d+", " ")]
        
        for old, new in replacements:
            cleand_message = re.sub(old, new, translated)
    
        message = [cleand_message]

        # vectorize the cleaned text
        message_vectorized = vectorizer.transform(message)

        # Predict the vectorized message
        EI_pred = EI_model.predict(message_vectorized)
        NS_pred = NS_model.predict(message_vectorized)
        FT_pred= FT_model.predict(message_vectorized)
        JP_pred = JP_model.predict(message_vectorized)

        # convert the prediction result from 1 and 0 to letters
        EI_output = 'E' if EI_pred == 0 else "I"
        NS_output = 'N' if NS_pred == 0 else "S"
        FT_output = 'F' if FT_pred == 0 else "T"
        JP_output = 'J' if JP_pred == 0 else "P"

        # MBTI Character's Message Type
        Type = EI_output + NS_output + FT_output + JP_output

        new_message = Message(message=message_content,
                            translated_message=translated,
                            cleaned_message=cleand_message, 
                            mbti_type=Type)

        try:
            db.session.add(new_message)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your message to the database.'
        
    else:
        messages = Message.query.order_by(Message.id).all()
        num_messages = Message.query.count()  # Counting the number of messages\

        type_dict = {}
        for message in messages:
            if message.mbti_type not in type_dict.keys():
                type_dict.__setitem__(message.mbti_type, 1)
            else:
                type_dict[message.mbti_type] += 1

        types = list(type_dict.keys())
        counts = list(type_dict.values())

        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({absolute:d})"

        wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: func(pct, counts),
                                        textprops=dict(color="w"))
        ax.legend(wedges, types,
                title="types",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))


        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title("MBTI Type of User")
        img_io = io.BytesIO()
        plt.savefig(img_io, format="png")
        img_io.seek(0)


        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({absolute:d})"
        wedges, texts, autotexts = ax.pie(counts,  autopct=lambda pct: func(pct, counts), wedgeprops=dict(width=0.5), startangle=-40)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(types[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)

        ax.set_title("User's MBTI Type")

        img_io2 = io.BytesIO()
        plt.savefig(img_io2, format="png")
        img_io2.seek(0)


    
        return render_template('index.html', messages=messages, num_messages=num_messages, plot_data=base64.b64encode(img_io.read()).decode(),
                               plot_data2=base64.b64encode(img_io2.read()).decode())
                    

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)