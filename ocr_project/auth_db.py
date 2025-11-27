"""User authentication database module using SQLite."""

import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path

# Database file path
DB_PATH = Path(__file__).parent / 'users.db'

def init_db():
    """Initialize the user authentication database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ User database initialized at {DB_PATH}")

def register_user(username, email, password):
    """
    Register a new user.
    
    Args:
        username: Unique username
        email: Unique email address
        password: Plain text password (will be hashed)
    
    Returns:
        dict: {'success': bool, 'message': str, 'user_id': int (if success)}
    """
    try:
        # Validate inputs
        if not username or not email or not password:
            return {'success': False, 'message': 'All fields are required'}
        
        if len(username) < 3:
            return {'success': False, 'message': 'Username must be at least 3 characters'}
        
        if len(password) < 6:
            return {'success': False, 'message': 'Password must be at least 6 characters'}
        
        if '@' not in email or '.' not in email:
            return {'success': False, 'message': 'Invalid email format'}
        
        # Hash password
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Insert user
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            
            return {
                'success': True,
                'message': 'Registration successful',
                'user_id': user_id
            }
            
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'username' in str(e):
                return {'success': False, 'message': 'Username already exists'}
            elif 'email' in str(e):
                return {'success': False, 'message': 'Email already registered'}
            else:
                return {'success': False, 'message': 'Registration failed'}
                
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def login_user(username_or_email, password):
    """
    Validate user login credentials.
    
    Args:
        username_or_email: Username or email address
        password: Plain text password
    
    Returns:
        dict: {'success': bool, 'message': str, 'user': dict (if success)}
    """
    try:
        if not username_or_email or not password:
            return {'success': False, 'message': 'All fields are required'}
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if input is email or username
        if '@' in username_or_email:
            cursor.execute('SELECT * FROM users WHERE email = ?', (username_or_email,))
        else:
            cursor.execute('SELECT * FROM users WHERE username = ?', (username_or_email,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return {'success': False, 'message': 'Invalid credentials'}
        
        # Verify password
        if not check_password_hash(user['password_hash'], password):
            conn.close()
            return {'success': False, 'message': 'Invalid credentials'}
        
        # Update last login
        cursor.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now(), user['id'])
        )
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'created_at': user['created_at']
            }
        }
        
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def get_user_by_id(user_id):
    """Get user information by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'created_at': user['created_at']
            }
        return None
        
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

# Initialize database on module import
if not DB_PATH.exists():
    init_db()
