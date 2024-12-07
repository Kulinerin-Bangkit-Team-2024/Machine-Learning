FROM node:20-slim

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 8000

RUN apt-get update -y && apt-get install -y openssl

RUN npx prisma generate

CMD ["npm", "start"]