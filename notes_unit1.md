### ¿Qué son los LLMs?

Los **Large Language Models (LLMs)** son modelos de inteligencia artificial especializados en comprender y generar lenguaje humano. Están entrenados con grandes volúmenes de datos textuales y cuentan con millones o incluso miles de millones de parámetros. La mayoría de estos modelos están basados en la arquitectura **Transformer**, popularizada desde la introducción de BERT en 2018.

#### **Tipos de Transformers**
1. **Encoders**: Analizan texto y generan representaciones densas. Ejemplo: **BERT** (usado en clasificación de texto, búsqueda semántica).
2. **Decoders**: Generan texto a partir de una secuencia de entrada. Ejemplo: **Llama** de Meta (usado en generación de texto y chatbots).
3. **Seq2Seq (Encoder–Decoder)**: Combinan ambos mecanismos para tareas como traducción y resumen. Ejemplo: **T5, BART**.

#### **Modelos LLM más conocidos**
- **GPT-4** (OpenAI)
- **Llama 3** (Meta)
- **Gemma** (Google)
- **Deepseek-R1** (DeepSeek)
- **SmolLM2** (Hugging Face)
- **Mistral** (Mistral AI)

### **Funcionamiento de los LLMs**
Los LLMs operan bajo un principio clave: **predicción del próximo token**, es decir, dado un contexto de texto previo, predicen la siguiente unidad de información. Estos tokens no siempre son palabras completas, sino fragmentos de palabras, lo que optimiza su eficiencia.

#### **Decodificación y tokens especiales**
Los LLMs son **autoregresivos**, es decir, generan texto de manera iterativa hasta alcanzar un token especial llamado **EOS (End of Sequence)**, que varía según el modelo. Ejemplo:
- **GPT-4**: `<|endoftext|>`
- **Llama 3**: `<|eot_id|>`
- **Gemma**: `<end_of_turn>`

### **Estrategias de Decodificación**
- **Greedy Decoding**: Selecciona siempre el token con mayor puntuación.
- **Beam Search**: Evalúa múltiples secuencias para encontrar la mejor combinación de palabras.

### **Entrenamiento de los LLMs**
El entrenamiento de los LLMs se realiza en dos fases:
1. **Pre-entrenamiento**: Aprenden patrones lingüísticos a partir de grandes cantidades de datos mediante aprendizaje auto-supervisado.
2. **Fine-tuning**: Ajuste en tareas específicas como generación de código, traducción o asistencia conversacional.

### **Uso y Aplicaciones**
Los LLMs pueden ejecutarse de dos maneras:
1. **Localmente**, si se dispone de hardware potente.
2. **A través de APIs en la nube**, como Hugging Face Serverless Inference API.

### **LLMs en Agentes de IA**
Los LLMs son la base de los **AI Agents**, permitiéndoles interpretar instrucciones, mantener contexto, planificar tareas y decidir qué herramientas usar. En este curso se explorará en detalle cómo integrarlos en agentes inteligentes.

------
### **Mensajes y Tokens Especiales en LLMs**

#### **Conversaciones y Estructura en Modelos de Lenguaje**
Los usuarios interactúan con los LLMs a través de mensajes en interfaces de chat, pero en realidad, estos mensajes son convertidos en una única secuencia de tokens antes de ser procesados por el modelo. Esto significa que el modelo **no recuerda conversaciones previas**, sino que las lee completas en cada interacción.

Para estructurar estas interacciones, se utilizan **chat templates**, que formatean los mensajes del usuario y del asistente para que el modelo los entienda correctamente. Cada LLM usa su propia estructura y tokens especiales para diferenciar los roles dentro de una conversación.

---

### **Mensajes en LLMs**
1. **System Messages (Mensajes del sistema):** 
   - Definen el comportamiento del modelo (ejemplo: ser educado o rebelde).
   - También proporcionan información sobre herramientas y acciones disponibles.

2. **Conversaciones entre usuario y asistente:**
   - Se estructuran en una lista de mensajes con roles (`user` y `assistant`).
   - **Ejemplo en JSON:**
     ```json
     [
        {"role": "user", "content": "I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
        {"role": "user", "content": "It's ORDER-123"}
     ]
     ```
   - Antes de ser enviadas al LLM, estas conversaciones son convertidas en un **prompt** con tokens especiales.

---

### **Diferencias entre Modelos y Tokens Especiales**
Cada modelo tiene su propio formato de mensajes. Ejemplos:

1. **SmolLM2**:
   ```plaintext
   <|im_start|>user
   I need help with my order<|im_end|>
   ```
2. **Llama 3.2**:
   ```plaintext
   <|start_header_id|>user<|end_header_id|>
   I need help with my order<|eot_id|>
   ```

Los **tokens especiales** (EOS, delimitadores) ayudan a los modelos a interpretar cuándo termina un mensaje y empieza otro.

---

### **Modelos Base vs. Modelos Instruct**
- **Base Model:** Solo predicen el siguiente token sin instrucciones específicas.
- **Instruct Model:** Son ajustados para seguir instrucciones y mantener conversaciones estructuradas.

Para hacer que un modelo base funcione como un instruct model, es necesario usar **chat templates** que den formato a los mensajes.

---

### **Uso de Chat Templates en Transformers**
Los chat templates permiten transformar mensajes en prompts adecuados para cada modelo. En la librería `transformers`, esto se gestiona con la función `apply_chat_template()`, que automatiza la conversión.

Ejemplo en Python:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
Este prompt generado está listo para ser usado como entrada en el modelo.

---

### **Nota**
Los chat templates estructuran las conversaciones entre usuarios y modelos de lenguaje, asegurando la correcta interpretación de los mensajes. Además, permiten adaptar modelos base a un comportamiento más conversacional mediante formatos específicos.

---

### **¿Qué son las Herramientas (Tools) en los Agentes de IA?**

Las **herramientas (Tools)** son funciones que amplían las capacidades de un modelo de lenguaje (LLM). Un agente de IA no puede ejecutar acciones por sí solo, pero puede generar texto que invoque herramientas predefinidas. La aplicación (el agente) se encarga de interpretar estas invocaciones y ejecutar las funciones correspondientes.

---

### **¿Qué es una Herramienta en un LLM?**
Es una función con un objetivo específico que el modelo puede "utilizar" a través de texto estructurado. Ejemplos comunes incluyen:

| **Herramienta**       | **Descripción** |
|------------------------|----------------|
| **Búsqueda Web**       | Recupera información actualizada de internet. |
| **Generación de Imágenes** | Crea imágenes basadas en descripciones textuales. |
| **Recuperación de Información** | Accede a datos de fuentes externas. |
| **Interfaz de API** | Conecta el agente con servicios externos como GitHub o Spotify. |

Una herramienta debe complementar las capacidades de un LLM. Por ejemplo, un **modelo no es bueno en cálculos matemáticos**, pero si le proporcionamos una calculadora como herramienta, obtendrá resultados precisos.

---

### **¿Cómo Funcionan las Herramientas?**
Los LLMs **no pueden ejecutar herramientas directamente**. En su lugar:
1. Se les proporciona una descripción de la herramienta en el **System Message**.
2. El modelo genera un comando en texto para invocar la herramienta.
3. El agente interpreta este comando y ejecuta la herramienta real.
4. La respuesta de la herramienta se envía de vuelta al modelo para que genere la respuesta final.

Desde la perspectiva del usuario, parece que el modelo "ejecutó" la herramienta, cuando en realidad fue la aplicación la que lo hizo en segundo plano.

---

### **Definiendo una Herramienta**
Cada herramienta debe describirse con:
- **Nombre**
- **Descripción de su función**
- **Argumentos esperados**
- **Tipo de salida**

Ejemplo en Python de una herramienta de multiplicación:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```
**Descripción en texto para el LLM:**
```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

Cuando el LLM ve esta descripción en el prompt, sabe cómo invocar la herramienta correctamente.

---

### **Automatización con Decoradores**
Para evitar definir herramientas manualmente, se pueden usar decoradores en Python que extraen automáticamente el nombre, descripción y argumentos de una función:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

Esto genera automáticamente la descripción de la herramienta sin necesidad de escribirla a mano.

---

### **Importancia de las Herramientas**
Las herramientas permiten que los agentes de IA superen las limitaciones de los modelos de lenguaje, proporcionándoles acceso a información en tiempo real, cálculos avanzados y acciones especializadas. Su integración adecuada en los agentes mejora drásticamente su funcionalidad.

-----
### **Comprendiendo los Agentes de IA a través del Ciclo Pensamiento-Acción-Observación**

Los **Agentes de IA** trabajan en un ciclo continuo compuesto por tres fases:  

1. **Pensamiento (Thought):** El modelo analiza la consulta y decide el siguiente paso.  
2. **Acción (Action):** El agente invoca herramientas con los argumentos adecuados.  
3. **Observación (Observation):** Recibe la respuesta de la herramienta y ajusta su razonamiento.  

Este proceso sigue un **bucle** que se repite hasta que se alcanza el objetivo del agente.

---

### **Ejemplo: Alfred, el Agente del Clima**  
Un usuario pregunta: *"¿Cuál es el clima actual en Nueva York?"*  
El agente Alfred sigue el ciclo:  

1. **Pensamiento:** “Necesito obtener los datos del clima en Nueva York. Hay una herramienta disponible para ello.”  
2. **Acción:** Alfred llama a la herramienta `get_weather` con el argumento `{ "location": "New York" }`.  
3. **Observación:** Recibe la respuesta: *"Clima en Nueva York: parcialmente nublado, 15°C, 60% de humedad."*  
4. **Reflexión:** Ahora que tiene los datos, puede generar la respuesta final.  
5. **Acción Final:** Alfred responde al usuario con la información obtenida.  

---

### **Lecciones clave**
✔ **Los agentes iteran hasta completar su objetivo:** Si un paso falla, pueden intentarlo nuevamente.  
✔ **Integración de herramientas:** Permiten que el agente acceda a información en tiempo real.  
✔ **Adaptación dinámica:** Cada ciclo mejora la respuesta del agente basándose en nuevas observaciones.  

