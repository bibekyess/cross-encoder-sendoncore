from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = model.predict([
  ['Where are you from?', 'Nepal is a beautiful country.'],
  ['Where are you from?', 'I work in ALI'],
  ['Where are you from?', 'I am from Nepal.']
  ])
print(scores)
