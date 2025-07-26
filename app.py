from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from difflib import get_close_matches
from chat_data import chat_data


app = Flask(__name__)
CORS(app)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get("message", "").lower()
    if user_input in chat_data:
        return jsonify({"response": chat_data[user_input]})
    match = get_close_matches(user_input, chat_data.keys(), n=1, cutoff=0.5)
    if match:
        return jsonify({"response": chat_data[match[0]]})
    return jsonify({"response": "Sorry, I couldn't find a relevant answer. Visit https://www.tnesevai.tn.gov.in for more info."})

 


@app.route('/result')
def result():
    return render_template('result_page.html')

@app.route('/complaint')
def complaint():
    return render_template('complaint.html')  # not complaint_page.html unless you renamed the file

@app.route('/main')
def render_main():
    return render_template('main.html')


@app.route('/api/recommend_schemes', methods=['POST'])
def recommend_schemes():
    data = request.get_json()
    age = int(data.get('age', 0))
    occupation = data.get('occupation', '').lower()
    income = data.get('income', '').lower()
    state = data.get('state', '').lower()

    schemes = []

    if 'farmer' in occupation:
        schemes.append({
            'name': 'PM Kisan Yojana',
            'description': '₹6000/year to eligible farmers in 3 installments.',
            'eligibility': 'Small & marginal farmers with cultivable land.',
            'url': 'https://pmkisan.gov.in',
            'color': '#138808',
            'icon': 'bi-flower2'
        })

    if age >= 60:
        schemes.append({
            'name': 'Senior Citizen Welfare',
            'description': 'Benefits for senior citizens.',
            'eligibility': 'Age 60 years and above.',
            'url': 'https://example.com/senior',
            'color': '#db2444',
            'icon': 'bi-heart-fill'
        })

    if 'student' in occupation:
        schemes.append({
            'name': 'National Scholarship Portal',
            'description': 'Scholarship up to ₹50,000 for students.',
            'eligibility': 'Merit-based for higher education.',
            'url': 'https://scholarships.gov.in',
            'color': '#ff9933',
            'icon': 'bi-mortarboard-fill'
        })

    if income in ['low', 'economically weaker section', 'ewc']:
        schemes.append({
            'name': 'PMAY Housing Scheme',
            'description': 'Financial assistance for house construction.',
            'eligibility': 'Economically weaker sections & low income groups.',
            'url': 'https://pmaymis.gov.in',
            'color': '#215ea3',
            'icon': 'bi-house-door-fill'
        })

    return jsonify({'schemes': schemes})


if __name__ == '__main__':
    app.run(debug=True,port=5000)
