#!/usr/bin/env python3
"""
Script simple para ejecutar la aplicación Streamlit sin configuración de email
"""

import os
import sys
import subprocess

def main():
    print("🚀 Iniciando aplicación Streamlit...")
    print("📱 La aplicación estará disponible en: http://localhost:8501")
    print("⏹️  Para detener: Ctrl+C")
    print("")
    
    # Cambiar al directorio del proyecto
    #os.chdir('/home/nicole/atamo_pvstand')
    
    # Ejecutar Streamlit con configuración automática
    try:
        subprocess.run([
            'streamlit', 'run', 'streamlit_iv_app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true'
        ])
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
