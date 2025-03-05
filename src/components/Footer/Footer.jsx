import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer-container">
      <div className="footer-content">
        <div className="footer-section">
          <h3>DeepGuard</h3>
          <p>Combating Digital Fraud through Advanced Detection Systems</p>
        </div>
        
        <div className="footer-section">
          <h4>Contact</h4>
          <p>Email: contact@deepguard.ai</p>
          <div className="social-icons">
            <a href="#github">GitHub</a>
            <a href="#twitter">Twitter</a>
            <a href="#linkedin">LinkedIn</a>
          </div>
        </div>
      </div>
      
      <div className="footer-bottom">
        <p>© 2023 DeepGuard. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;