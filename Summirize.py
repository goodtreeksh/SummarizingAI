import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ConversationDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.dialogue = data['body']['dialogue']
            self.summary = data['body']['summary']
        
        # 전처리: 대화의 모든 발화를 하나의 문자열로 결합
        self.dialogue_text = ' '.join([utterance['utterance'] for utterance in self.dialogue])
    
    def __len__(self):
        return len(self.dialogue)
    
    def __getitem__(self, idx):
        return self.dialogue[idx]['utterance']

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, source):
        encoder_output, encoder_hidden = self.encoder(source)
        decoder_output, _ = self.decoder(encoder_output[-1].unsqueeze(0), encoder_hidden)
        output = self.fc(decoder_output)
        return output


# 데이터셋 로드
dataset = ConversationDataset('data.json')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 모델 초기화
input_size = 100  # 예시로 임의의 입력 크기 설정
hidden_size = 256
output_size = 100  # 예시로 임의의 출력 크기 설정
model = Seq2SeqModel(input_size, hidden_size, output_size)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        source = batch[0]  # 대화 발화를 입력으로 사용
        target = batch[0]  # 자기 자신을 타겟으로 사용 (간단한 Autoencoder 예시)
        
        # 입력 데이터 준비 (예시로 텐서로 변환)
        source_tensor = torch.tensor(source)
        
        # 순전파 및 손실 계산
        output = model(source_tensor)
        loss = criterion(output, source_tensor)
        
        # 역전파 및 옵티마이저 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

# 학습된 모델을 사용하여 대화를 요약하는 함수를 정의할 수 있습니다.
def summarize_conversation(model, conversation):
    # 대화 문장을 토큰화하고 정제하는 전처리 과정이 필요합니다.
    # 여기서는 단순히 예시로 처리하지만, 실제로는 더 복잡한 전처리 과정이 필요할 수 있습니다.
    tokens = conversation.split()
    input_tensor = torch.tensor(tokens)  # 예시로 토큰들을 텐서로 변환
    
    output_tensor = model(input_tensor)
    # 예측 결과를 텍스트로 변환하여 반환
    return ' '.join(output_tensor)
