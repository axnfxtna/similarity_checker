# similarity_checker

A Streamlit web app for check similarity of the documents with the vector database, by user input single pdf file and the web will show the percentage similarity in average, each page and with explanation.

**1. Features**
- Text-to-text similarity search  
- Vector database search using Milvus
- uses an LLM to generate similarity responses based on those documents
- Interactive results visualization in Streamlit

**2. Architecture**
- **System** : Ubuntu 24.04.2
- **Frontend**: Streamlit
- **Backend** : FastAPI
- **LLM** : ollama
- **Database** : Milvus-standalone(local)
- **Embedding Models** : all-MiniLM-L6-v
- **LLM Model** : qwen3:0.6b-q4_K_M
- **Configs** : Defined in `configs/configs.yaml`

**3. Installation**

   ```bash
   - clone repo
      git clone https://github.com/axnfxtna/similarity_checker.git
      cd similarity_checker
   - environment setup
      conda create -n ragdoc python=3.11
      conda activate ragdoc
      pip install -r requirements.txt
   ```
**4. Docker Installation**
  https://docs.docker.com/engine/install/ubuntu/
  ```bash
  - Set up Docker's apt repository.
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

  - Install the Docker packages
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
 ```
**5. Milvus Installation**
  - after pull this git
    ```bash
    cd milvus
    sudo docker network create ragdoc-network
    sudo docker compose up -d

    Creating milvus-etcd  ... done
    Creating milvus-minio ... done
    Creating milvus-standalone ... done
    ```


**6. Configuration**
- Create a .env file in the project root:
   ```ini
   URI="http://milvus-standalone:19530"
   ```
   - Model, DB, and embedding settings can also be modified in:
   ```bash
   configs/configs.yaml
   ```
   - If you want to use cpu to run all the process, edit docker-compose.yml
   ```yml
   ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    ports:
      - "11435:11434"
    environment:
      - OLLAMA_DEVICE=cpu          
      - OLLAMA_LISTEN_BACKLOG=6000
    ulimits:
      nofile:
        soft: 6000
        hard: 6000
    networks:
      - ragdoc-network
    volumes:
      - ollama_models:/root/.ollama
    ```
**7. Usage**
   - Create a .env file in the project root:
   ```ini
   URI="http://milvus-standalone:19530"
   ```
   - Model, DB, and embedding settings can also be modified in:
   ```bash
   configs/configs.yaml
   ```
   - If you want to use cpu to run all the process, edit docker-compose.yml
   ```yml
   ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    ports:
      - "11435:11434"
    environment:
      - OLLAMA_DEVICE=cpu          
      - OLLAMA_LISTEN_BACKLOG=6000
    ulimits:
      nofile:
        soft: 6000
        hard: 6000
    networks:
      - ragdoc-network
    volumes:
      - ollama_models:/root/.ollama
    ```


   - edit the configs, docker-compose, and Dockerfile to be your dataset

   ```bash
    - in configs
    collection_name: 'file'
    dataset_dir: '/app/dataset'
    - in Dockerfile_database, Dockerfile_streamlit, and Dockerfile_similarity
    COPY dataset /app/dataset
    - in docker-compose.yml at streamlit-app, similarity-app and database-app
    volumes:
      - /home/natcha/rag_document/dataset:/app/dataset
   - Using docker compose to run the model
     sudo docker compose build
     sudo docker compose up
   - set up ollama in docker
     sudo docker compose run similarity-app bash
     ollama pull qwen3:0.6b-q4_K_M
     exit
   - restart the container
     sudo docker compose down
     sudo docker compose up
  ```
   - Streamlit => URL: http://0.0.0.0:8501
   - API => Uvicorn running on http://0.0.0.0:8005
     
**8. Example**
  ```bash
    - input
      Upload your search PDF: pdf file
    - output from the check
      overall average percentage
      similarity percentage of query page pdf and match pdf
    - output from the explanation
      overall average percentage
      similarity percentage of query page pdf and match pdf
      explanation how the query page and match pdf have similarity percentage
      
  ```