# Guide de soutenance — Assistant IA AGS GROUP

## 1. Le pitch (30-45 secondes, à dire par cœur)

"Pendant mon stage chez AGS GROUP, entreprise de construction de villas et riads
à Marrakech, j'ai constaté que toute l'information de l'entreprise — devis types,
normes, cahiers des charges, réponses aux clients — était traitée manuellement
et dispersée dans des fichiers non centralisés. J'ai conçu et développé un
assistant IA basé sur une architecture RAG (Retrieval-Augmented Generation) qui
permet d'interroger en langage naturel toute la connaissance de l'entreprise,
avec des réponses fiables et sourcées, disponibles 24h/24."

## 2. Pourquoi RAG et pas juste ChatGPT ?
Argument clé à donner au jury :
- Un LLM générique (ChatGPT) **invente** des réponses ("hallucinations") et ne
  connaît pas les documents internes d'AGS GROUP.
- Le RAG **ancre** les réponses dans les vrais documents de l'entreprise : le
  modèle ne répond qu'à partir de ce qu'il a retrouvé dans la base de
  connaissances, et **cite ses sources**. C'est indispensable dans un métier
  où une erreur sur un délai ou une garantie a un impact commercial et légal.

## 3. Architecture à expliquer (dessine-la au tableau si possible)
```
Question utilisateur
     │
     ▼
Recherche vectorielle (Chroma) ──► Documents pertinents (cahier des charges, FAQ, normes...)
     │
     ▼
LLM (Llama 3.3 via Groq) + contexte trouvé
     │
     ▼
Réponse + sources citées
```
Vocabulaire à maîtriser : **embeddings**, **chunking**, **base vectorielle**,
**retrieval**, **prompt engineering**, **hallucination**.

## 4. Valeur business pour AGS GROUP
- Réponse instantanée aux clients, disponible en dehors des heures de bureau.
- Uniformisation de l'information donnée (plus de divergence entre commerciaux).
- Formation accélérée des nouveaux employés (ils interrogent l'assistant au
  lieu d'apprendre "sur le tas").
- Base extensible : demain on peut y ajouter les devis réels, l'historique
  client, un module de génération automatique de devis.

## 5. Limites (à assumer, ça montre ta maturité technique)
- La qualité des réponses dépend de la qualité des documents fournis en entrée
  ("garbage in, garbage out").
- Le prototype actuel utilise des documents d'exemple ; en production il
  faudrait l'alimenter avec les vrais documents internes d'AGS GROUP.
- Pas encore de gestion des utilisateurs / droits d'accès selon les rôles.

## 6. Perspectives d'évolution (à mentionner à la fin, ça impressionne)
- Intégration WhatsApp Business pour que les clients posent leurs questions
  directement depuis WhatsApp.
- Ajout d'un module de génération automatique de devis à partir d'un
  formulaire (surface, type de bien, style).
- Suivi de chantier avec notifications automatiques aux clients.
- Passage à un LLM multilingue (français / arabe / darija) pour coller aux
  usages réels de la clientèle.

## 7. Questions probables du jury + réponses courtes

**Q: Pourquoi Chroma et pas une autre base vectorielle ?**
R: Chroma est open-source, légère, fonctionne en local sans infrastructure
supplémentaire — adapté à un prototype et à une PME comme AGS GROUP.

**Q: Pourquoi Groq/Llama et pas OpenAI/GPT ?**
R: Groq propose une inférence très rapide et un accès gratuit pour prototyper,
avec un modèle open-weight performant (Llama 3.3 70B). L'architecture reste
compatible avec OpenAI si l'entreprise préfère cette option en production.

**Q: Comment garantissez-vous que le chatbot ne donne pas de fausses informations ?**
R: Grâce au RAG : le prompt impose au modèle de répondre uniquement à partir
du contexte récupéré, et de dire explicitement qu'il ne sait pas si
l'information n'est pas dans les documents.

**Q: Combien ça coûte à faire tourner ?**
R: L'indexation et les embeddings sont gratuits (modèle local). L'appel au LLM
Groq est gratuit dans les limites du tier gratuit, avec un coût très faible
en production vu le volume de requêtes d'une PME.

**Q: Comment mesureriez-vous la performance du système ?**
R: Taux de réponses correctes sur un jeu de questions test, temps de réponse
moyen, taux de questions sans réponse trouvée (pour identifier les trous dans
la base documentaire).

## 8. Déroulé conseillé de la démo live (3-4 minutes)
1. Montre l'interface, explique en une phrase ce que c'est.
2. Pose une question simple ("Quel est le délai pour une villa de 300m² ?").
3. Montre la réponse ET les sources (clique sur l'expander).
4. Pose une question hors-sujet ou sans réponse dans les docs, pour montrer
   que le système dit honnêtement "je ne sais pas" (ex: "Quelle est la couleur
   préférée du PDG ?") — ça prouve que le système ne hallucine pas.
5. Termine sur les perspectives d'évolution.
