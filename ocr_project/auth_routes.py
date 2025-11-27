# Authentication routes - add these at the end of app.py before if __name__ == '__main__':

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page and authentication."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = login_user(username, password)
        
        if result['success']:
            session.permanent = True
            session['user_id'] = result['user']['id']
            session['username'] = result['user']['username']
            session['email'] = result['user']['email']
            
            return jsonify({'success': True, 'message': 'Login successful', 'user': result['user']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during login'}), 500

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    """Registration page and user creation."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('register.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        result = register_user(username, email, password)
        
        if result['success']:
            return jsonify({'success': True, 'message': 'Registration successful'})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during registration'}), 500

@app.route('/logout')
def logout():
    """Logout user and clear session."""
    session.clear()
    return redirect(url_for('login_page'))
