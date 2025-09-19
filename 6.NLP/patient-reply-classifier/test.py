import requests
import json

# url = "https://test-detect-out-of-place-open-responses"

payload = json.dumps({
                "asr": [
                    "Sí soy yo",
                    "Igual que siempre",
                    "Sí siempre aunque fui al médico y me cambiaron la medicación",
                    "No no nunca",
                    "No todo igual que siempre",
                    "No",
                    "No nada",
                    "Sí",
                    "No ninguno",
                    "Nada todo bien pero si es verdad que me duele mucho la cadera",
                    "Eh no",
                    "Sí",
                    "Pues mira, lo que pasa es que me duele bastante la cabeza y he tenido que cambiar la medicación y no sé si me tienen que volver a llamar o tengo que acercarme yo al."
                ],
                "intent": [
                    "Welcome - yes",
                    "WellbeingShort - same",
                    "medicationDaily - yes",
                    "medicationForget - no",
                    "medicationChange - no",
                    "changeMeals - no",
                    "homeopathicTreatment - no",
                    "TRT - yes",
                    "Bleeding - no",
                    "ictusSymptoms - fallback",
                    "Emergency - no",
                    "Bye - yes",
                    "Bye - yes - open"
                ]
            })

headers = {
  "Content-Type": "application/json"
}

response = requests.request("POST", url, headers=headers, data=payload, timeout=300)

print(response.text)
print(response.status_code)