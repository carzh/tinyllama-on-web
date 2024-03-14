#! /bin/bash

cp /c/Users/carolinezhu/Documents/transformers.js/xenova-transformers-2.15.0.tgz .

rm -rf node_modules
rm package-lock.json

npm cache clean --force
npm cache verify

npm install
npm run dev
