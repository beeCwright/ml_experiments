# Generate Root Certiicate and Root Key
openssl req -out rootCA.pem -keyout rootKey.pem -new -x509 -days 3650 -nodes -subj "/C=US/ST=MA/O=CCDS/CN=root"

# Generate Server key and Signed Certificates
echo "00" > file.srl
openssl genrsa -out server.key 2048
openssl req -key server.key -new -out server.req -subj  "/C=US/ST=MA/O=CCDS/CN=d7920-12.ccds.io"
openssl x509 -req -in server.req -CA rootCA.pem -CAkey rootKey.pem -CAserial file.srl -out server.crt -days 3650
cat server.key server.crt > server.pem
openssl verify -CAfile rootCA.pem server.pem

# Generate Client Key and Signed Certificates
openssl genrsa -out client.key 2048
openssl req -key client.key -new -out client.req -subj "/C=US/ST=MA/O=CCDS/CN=d7920-12.ccds.io"
openssl x509 -req -in client.req -CA rootCA.pem -CAkey rootKey.pem -CAserial file.srl -out client.crt -days 3650
cat client.key client.crt > client.pem
openssl verify -CAfile rootCA.pem client.pem

mongod --bind_ip_all --sslMode requireSSL --sslPEMKeyFile server.pem --sslCAFile rootCA.pem