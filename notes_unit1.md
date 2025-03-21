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
----
### **Razonamiento Interno y el Enfoque ReAct en los Agentes de IA**  

El **razonamiento interno (Thought)** de un agente de IA es su capacidad para analizar información, descomponer problemas complejos en pasos más manejables y decidir qué acción tomar a continuación. Este proceso permite a los modelos **ajustar sus planes en función de nueva información y reflexionar sobre decisiones previas**.

Los pensamientos pueden incluir:  
✔ **Planificación:** Definir pasos para resolver una tarea.  
✔ **Análisis:** Identificar problemas y evaluar soluciones.  
✔ **Toma de decisiones:** Elegir la mejor acción basada en restricciones.  
✔ **Resolución de problemas:** Optimizar estrategias para mejorar resultados.  
✔ **Integración de memoria:** Recordar preferencias del usuario.  
✔ **Autorreflexión:** Evaluar enfoques pasados y corregir errores.  
✔ **Priorización:** Determinar qué tareas son más urgentes.  

---

### **El Enfoque ReAct: Razonar antes de Actuar**  
**ReAct (Reasoning + Acting)** es una técnica de prompting que **fomenta que el modelo piense paso a paso antes de actuar**.  
✔ En lugar de generar una respuesta inmediata, el LLM primero analiza el problema y elabora un plan.  
✔ Se implementa con prompts como *"Let’s think step by step"*, lo que reduce errores y mejora la calidad de la respuesta.  

Además, modelos como **Deepseek R1 y OpenAI’s o1** han sido entrenados para incluir automáticamente secciones de pensamiento antes de responder, utilizando tokens especiales `<think>...</think>`.  

---

## **Técnicas de Prompting: Few-Shot, Few-Shot-CoT, Zero-Shot y Zero-Shot-CoT (Ours)**  

### **1️⃣ Zero-Shot**  
✔ El modelo recibe la pregunta sin ejemplos previos.  
✔ Responde directamente basándose en su entrenamiento.  
✔ **Ejemplo:**  
   **Prompt:** *“¿Cuál es la capital de Francia?”*  
   **Respuesta:** *“París”*  

### **2️⃣ Zero-Shot-CoT (Chain of Thought - Ours)**  
✔ Similar a Zero-Shot, pero el modelo es instruido para razonar antes de responder.  
✔ Se usa un prompt como *"Explícame paso a paso antes de dar la respuesta."*  
✔ **Ejemplo:**  
   **Prompt:** *“¿Cuántas patas tienen tres gatos en total?”*  
   **Respuesta:**  
   1. Un gato tiene 4 patas.  
   2. Tres gatos tienen 3 × 4 = 12 patas.  
   3. Respuesta final: *12 patas.*  

### **3️⃣ Few-Shot**  
✔ Se proporcionan algunos ejemplos antes de la pregunta principal.  
✔ El modelo aprende del patrón y lo aplica a nuevas consultas.  
✔ **Ejemplo:**  
   **Prompt:**  
   - *Ejemplo 1: Un perro tiene 4 patas. Un gato tiene 4 patas.*  
   - *Ejemplo 2: Un caballo tiene 4 patas.*  
   - *Pregunta: ¿Cuántas patas tiene un elefante?*  
   **Respuesta:** *4 patas*  

### **4️⃣ Few-Shot-CoT (Chain of Thought)**  
✔ Combina ejemplos con razonamiento paso a paso.  
✔ Mejora la precisión en problemas más complejos.  
✔ **Ejemplo:**  
   **Prompt:**  
   - *Ejemplo 1:* "Si un tren recorre 60 km en una hora, en 3 horas recorrerá 180 km."  
   - *Ejemplo 2:* "Si un coche viaja a 50 km/h, en 4 horas recorrerá 200 km."  
   - *Pregunta:* "Si un avión viaja a 800 km/h, ¿cuánto recorrerá en 2 horas?"  
   **Respuesta:**  
   1. El avión viaja a 800 km/h.  
   2. En 2 horas recorrerá 800 × 2 = 1600 km.  
   3. **Respuesta final:** *1600 km*  

---

### **Notas**  
El uso de **ReAct** y **Chain of Thought (CoT)** mejora la capacidad de los agentes de IA para **razonar antes de responder**, lo que resulta en respuestas más precisas y estructuradas. **Zero-Shot y Few-Shot** permiten ajustar el nivel de contexto necesario, con **Zero-Shot-CoT y Few-Shot-CoT** ofreciendo las mejores soluciones para tareas complejas.
----
### **Acciones en los Agentes de IA**  

Las **acciones** son los pasos concretos que un agente de IA realiza para interactuar con su entorno, como buscar información, llamar APIs o controlar dispositivos. Para ejecutarlas, los agentes pueden representar las acciones en **JSON, código o llamadas a funciones**.

---

### **Tipos de Agentes según su Ejecución de Acciones**  

| **Tipo de Agente**       | **Descripción** |
|------------------------|----------------|
| **JSON Agent**         | Genera acciones en formato JSON. |
| **Code Agent**         | Escribe código ejecutable para realizar acciones. |
| **Function-calling Agent** | Subcategoría del JSON Agent que genera un nuevo mensaje estructurado para cada acción. |

Las acciones pueden clasificarse en:  

✔ **Recopilación de información:** Consultas web, bases de datos, recuperación de documentos.  
✔ **Uso de herramientas:** Llamadas a API, cálculos, ejecución de código.  
✔ **Interacción con el entorno:** Control de interfaces digitales o dispositivos físicos.  
✔ **Comunicación:** Chat con usuarios o colaboración entre agentes.  

---

### **El Enfoque "Stop and Parse"**  

Para evitar errores y salidas no deseadas, los agentes deben **detener la generación de tokens una vez completada una acción**.  

1. **Generación en un formato estructurado** (JSON o código).  
2. **Detención de la generación** tras definir la acción.  
3. **Parseo de la salida** para extraer la función a ejecutar y los parámetros requeridos.  

#### **Ejemplo: JSON Agent llamando a una API del clima**  
```json
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```
Esto permite que el framework identifique la función `get_weather` y extraiga el parámetro `"New York"`.

---

### **Code Agents: Generación de Código Ejecutable**  

En lugar de JSON, un **Code Agent** genera código en un lenguaje como Python.  

✔ **Mayor expresividad:** Puede incluir bucles, condicionales y lógica más avanzada.  
✔ **Modularidad:** Puede reutilizar funciones en varias tareas.  
✔ **Depuración más sencilla:** Errores en el código son más fáciles de detectar.  
✔ **Integración directa:** Permite conectar con bibliotecas y APIs.  

#### **Ejemplo: Code Agent llamando a una API del clima en Python**  
```python
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Ejecutar la función y mostrar el resultado
result = get_weather("New York")
print(f"The current weather in New York is: {result}")
```
**Flujo del Code Agent:**  
1. Genera código que llama a la API.  
2. Procesa la respuesta.  
3. Imprime el resultado como salida final.  

---

### **Notas**  
✔ **Los agentes de IA ejecutan acciones en distintos formatos** (JSON, código o llamadas a funciones).  
✔ **El método "Stop and Parse" garantiza respuestas precisas y estructuradas.**  
✔ **Los Code Agents ofrecen mayor flexibilidad, pero requieren ejecución externa.**  
-----

### **Observe – Integrando Retroalimentación para Reflexionar y Adaptarse**

La fase **Observe** es cuando el **Agente de IA percibe las consecuencias de sus acciones**. Las observaciones proporcionan retroalimentación valiosa que impulsa el razonamiento del agente y guía sus futuras decisiones.

---

### **¿Qué sucede en esta fase?**

1. **Recopila Retroalimentación:**  
   Recibe datos que indican si la acción fue exitosa o fallida (por ejemplo, la respuesta de una API).

2. **Actualiza el Contexto:**  
   La observación se añade a la "memoria" del agente (es decir, al prompt), para mantenerlo informado del estado actual.

3. **Adapta su Estrategia:**  
   El agente usa esta nueva información para ajustar su razonamiento y decidir los siguientes pasos.

---

### **Ejemplo:**
Si un agente llama a una API del clima y recibe:  
*“parcialmente nublado, 15°C, 60% de humedad”*,  
esa respuesta se convierte en una **observación**, y se integra al prompt como nuevo contexto. Con ello, el agente puede decidir si ya tiene lo necesario para responder al usuario o si necesita realizar otra acción.

---

### **Tipos de Observaciones**

| **Tipo**             | **Ejemplo**                                                |
|----------------------|------------------------------------------------------------|
| **Retroalimentación del sistema** | Mensajes de error, códigos de estado, confirmaciones. |
| **Cambios de datos**             | Actualizaciones en bases de datos, archivos, o estados. |
| **Datos del entorno**            | Lecturas de sensores, métricas del sistema.             |
| **Análisis de respuestas**       | Respuestas de APIs, resultados de cálculos.             |
| **Eventos basados en tiempo**    | Tareas programadas completadas, plazos alcanzados.      |

---

### **¿Cómo se integran las observaciones?**

1. El agente genera una acción.  
2. El entorno (framework) **ejecuta la acción**.  
3. El resultado se **añade como observación** al prompt.  
4. El agente **reanuda su razonamiento**, incorporando ese nuevo contexto.

---

### **Nota**

La fase **Observe** permite que los agentes sean **adaptativos, reflexivos y dinámicos**. Al integrar retroalimentación constantemente, el agente puede **afinar su comportamiento, corregir errores y tomar mejores decisiones** en cada ciclo.
