#!/bin/bash
# 학습 진행 상황 모니터링 스크립트

LOG_DIR="logs/rsl_rl/hierarchical_nav"
LATEST_RUN=$(ls -t $LOG_DIR | head -1)
FULL_PATH="$LOG_DIR/$LATEST_RUN"

echo "=========================================="
echo "학습 진행 상황 모니터링"
echo "=========================================="
echo "최신 실행: $LATEST_RUN"
echo "전체 경로: $FULL_PATH"
echo ""

# 체크포인트 파일 확인
echo "--- 체크포인트 파일 ---"
ls -lh $FULL_PATH/*.pt 2>/dev/null | tail -5
echo ""

# TensorBoard 이벤트 파일 확인
echo "--- TensorBoard 이벤트 파일 ---"
if [ -f $FULL_PATH/events.out.tfevents.* ]; then
    echo "✅ TensorBoard 로그 파일이 있습니다"
    echo "TensorBoard 실행: tensorboard --logdir=$LOG_DIR --port=6006"
else
    echo "⚠️  TensorBoard 로그 파일이 아직 생성되지 않았습니다"
fi
echo ""

# 설정 파일 확인
echo "--- 설정 파일 ---"
if [ -d $FULL_PATH/params ]; then
    echo "✅ 설정 파일 디렉토리가 있습니다"
    ls $FULL_PATH/params/
else
    echo "⚠️  설정 파일 디렉토리가 없습니다"
fi
echo ""

# 최신 체크포인트 정보
LATEST_CHECKPOINT=$(ls -t $FULL_PATH/model_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "--- 최신 체크포인트 ---"
    echo "파일: $(basename $LATEST_CHECKPOINT)"
    echo "크기: $(du -h $LATEST_CHECKPOINT | cut -f1)"
    echo "수정 시간: $(stat -c %y $LATEST_CHECKPOINT)"
fi
echo ""

echo "=========================================="
echo "TensorBoard 실행 방법:"
echo "  tensorboard --logdir=$LOG_DIR --port=6006"
echo "  그 다음 브라우저에서 http://localhost:6006 접속"
echo "=========================================="

