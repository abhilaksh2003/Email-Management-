from flask import Flask, request, render_template
from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Configure SQLAlchemy part of the app instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emails.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create SQLAlchemy db instance
db = SQLAlchemy(app)

# Define the Email model
class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)

# Augmented sample data (unchanged)
emails = [
    "Meeting with the client to discuss project updates",
    "Lunch with family this weekend",
    "50% off on your next purchase!",
    "Your account statement is now available",
    "Team outing this Friday",
    "Exclusive offer just for you!",
    "Project deadline extended",
    "Birthday party invitation",
    "Weekly newsletter from your favorite store",
    "Reminder: Doctor's appointment tomorrow",
    "Quarterly business review meeting",
    "Family picnic next Saturday",
    "Special discount on electronics!",
    "Your credit card bill is ready",
    "Office party this Thursday",
    "Limited time offer for loyal customers!",
    "Final project submission date",
    "Wedding anniversary celebration",
    "Monthly deals from your favorite retailer",
    "Don't forget your dentist appointment next week"
]

labels = [
    "work",
    "personal",
    "promotions",
    "updates",
    "work",
    "promotions",
    "work",
    "personal",
    "promotions",
    "updates",
    "work",
    "personal",
    "promotions",
    "updates",
    "work",
    "promotions",
    "work",
    "personal",
    "promotions",
    "updates"
]


# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Global list to store predictions
predictions = []

# Function to classify new emails and store in database
@app.route('/classify', methods=['POST'])
def classify():
    email = request.form['email']
    category = classify_email(email)
    
    # Store the email and its category in the database
    new_email = Email(email_text=email, category=category)
    db.session.add(new_email)
    db.session.commit()
    
    # Append prediction to global predictions list
    predictions.append({'email': email, 'category': category})
    
    return render_template('result.html', email=email, category=category)

# Function to classify new emails using the trained model
def classify_email(email):
    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)
    return prediction[0]

# Function to display all stored emails from the database
@app.route('/database')
def show_database():
    # Query all stored emails from the database
    emails = Email.query.all()
    return render_template('database.html', emails=emails)

# Function to render index page
@app.route('/')
def index():
    return render_template('index.html')

# Function to render predictions page
@app.route('/predictions')
def view_predictions():
    # Initialize empty dictionaries to hold categorized predictions
    categorized_predictions = {'work': [], 'personal': [], 'promotions': [], 'updates': []}

    # Categorize predictions based on their category
    for prediction in predictions:
        category = prediction['category']
        if category in categorized_predictions:
            categorized_predictions[category].append(prediction)

    return render_template('predictions.html', categorized_predictions=categorized_predictions)

@app.route('/reset_database', methods=['POST'])
def reset_database():
    try:
        db.drop_all()
        db.create_all()
        return redirect(url_for('show_database'))
    except Exception as e:
        print(str(e))
        return redirect(url_for('show_database'))
    
@app.route('/delete/<int:id>', methods=['POST'])
def delete_email(id):
    email_to_delete = Email.query.get_or_404(id)
    try:
        db.session.delete(email_to_delete)
        db.session.commit()
        return redirect(url_for('show_database'))
    except Exception as e:
        print(str(e))
        return redirect(url_for('show_database'))
        return 'Error deleting email'

      
    

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create all tables defined in models
    app.run(debug=True)
