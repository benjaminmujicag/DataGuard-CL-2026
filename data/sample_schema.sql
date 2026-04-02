-- ==============================================================================
-- CL-DataGuard 2026 - Esquema de Base de Datos Clínico-Financiera (Ficticio)
-- ==============================================================================
--
-- NOTA: Este archivo contiene únicamente instrucciones DDL (Estructuras de tabla)
-- No contiene, ni debe contener, sentencias DML (INSERT) ni datos de usuarios reales.
-- Este archivo es de uso exclusivo para las pruebas de lectura de metadata del Agente.

-- 1. Tabla de Usuarios (Contiene Datos Identificadores)
CREATE TABLE IF NOT EXISTS usuarios (
    usuario_id INT PRIMARY KEY AUTO_INCREMENT,
    rut VARCHAR(12) NOT NULL UNIQUE,       -- Dato personal directo
    nombres VARCHAR(100) NOT NULL,         -- Dato personal
    apellidos VARCHAR(100) NOT NULL,       -- Dato personal
    fecha_nacimiento DATE,                 -- Dato personal
    email VARCHAR(150),                    -- Dato de contacto
    telefono VARCHAR(15),                  -- Dato de contacto
    direccion_postal VARCHAR(255),         -- Dato personal / Contacto / Geolocalización
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Tabla Clínica (Contiene Datos Altamente Sensibles)
CREATE TABLE IF NOT EXISTS fichas_clinicas (
    ficha_id INT PRIMARY KEY AUTO_INCREMENT,
    usuario_id INT NOT NULL,
    diagnostico_medico TEXT,               -- Dato SENSIBLE (Salud)
    tratamiento_psiquiatrico BOOLEAN,      -- Dato SENSIBLE (Salud mental)
    grupo_sanguineo VARCHAR(5),            -- Dato SENSIBLE (Salud)
    orientacion_sexual VARCHAR(50),        -- Dato SENSIBLE (Categoría especial)
    etnia_origen VARCHAR(50),              -- Dato SENSIBLE (Categoría especial)
    FOREIGN KEY (usuario_id) REFERENCES usuarios(usuario_id)
);

-- 3. Tabla Biométrica y Seguridad (Contiene Datos Sensibles y Logs)
CREATE TABLE IF NOT EXISTS control_accesos (
    acceso_id INT PRIMARY KEY AUTO_INCREMENT,
    usuario_id INT NOT NULL,
    huella_dactilar_hash VARCHAR(255),     -- Dato SENSIBLE (Biométrico)
    reconocimiento_facial_vector BLOB,     -- Dato SENSIBLE (Biométrico)
    ip_conexion VARCHAR(45),               -- Dato inofensivo / log bajo
    ultima_conexion TIMESTAMP,             -- Log inofensivo
    user_agent_navegador VARCHAR(255)      -- Log inofensivo
);

-- 4. Tabla Financiera (Contiene Datos Financieros Sensibles)
CREATE TABLE IF NOT EXISTS informacion_financiera (
    cuenta_id INT PRIMARY KEY AUTO_INCREMENT,
    usuario_id INT NOT NULL,
    sueldo_liquido DECIMAL(10,2),          -- Dato Financiero (Alto riesgo)
    numero_tarjeta_credito VARCHAR(20),    -- Dato Financiero crítico
    banco_institucion VARCHAR(100),        -- Dato Financiero
    historial_embargos BOOLEAN,            -- Dato Financiero Sensible
    FOREIGN KEY (usuario_id) REFERENCES usuarios(usuario_id)
);

-- 5. Tabla de Configuración de Sistema (Datos inofensivos / Sin riesgo)
CREATE TABLE IF NOT EXISTS configuraciones_globales (
    config_param VARCHAR(100) PRIMARY KEY,
    config_valor VARCHAR(255) NOT NULL,    -- Cero riesgo, metadato de app
    ultima_actualizacion TIMESTAMP,        -- Cero riesgo
    mantenimiento_activo BOOLEAN           -- Cero riesgo
);
