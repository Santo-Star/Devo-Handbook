@echo off
echo ==========================================
echo   ☕ INICIANDO DEVOCION AI - PDF CHAT
echo ==========================================
echo.

:: Verificar si existe Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta en el PATH o no esta instalado.
    echo Por favor, instala Python 3.10+ desde python.org
    pause
    exit /b
)

:: Instalar dependencias si es necesario
echo [1/2] Verificando e instalando dependencias (esto puede tardar la primera vez)...
:: No redirigir pip a nul para que el usuario o nosotros podamos ver que esta pasando
python -m pip install -r requirements.txt --upgrade

if %errorlevel% neq 0 (
    echo [ERROR] Hubo un problema instalando las dependencias.
    echo Revisa los mensajes de arriba para identificar el problema de red o de permisos.
    pause
    exit /b
)

:: Iniciar Streamlit
echo [2/2] Lanzando servidor de aplicacion local...
echo.
python -m streamlit run app.py

pause
