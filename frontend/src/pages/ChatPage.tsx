import React, { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import InputChatContent from '../components/InputChatContent';
import useChat from '../hooks/useChat';
import ChatMessage from '../components/ChatMessage';
import useScroll from '../hooks/useScroll';
import { useNavigate, useParams } from 'react-router-dom';
import { PiArrowsCounterClockwise, PiLink, PiPencilLine, PiStar, PiStarFill, PiWarningCircleFill } from 'react-icons/pi';
import Button from '../components/Button';
import { useTranslation } from 'react-i18next';
import SwitchBedrockModel from '../components/SwitchBedrockModel';
import useBot from '../hooks/useBot';
import useConversation from '../hooks/useConversation';
import ButtonPopover from '../components/PopoverMenu';
import PopoverItem from '../components/PopoverItem';
import { copyBotUrl } from '../utils/BotUtils';
import { produce } from 'immer';
import ButtonIcon from '../components/ButtonIcon';
import StatusSyncBot from '../components/StatusSyncBot';
import Alert from '../components/Alert';
import useBotSummary from '../hooks/useBotSummary';
import useModel from '../hooks/useModel';
import { io } from 'socket.io-client';

const MISTRAL_ENABLED: boolean = import.meta.env.VITE_APP_ENABLE_MISTRAL === 'true';

const ChatPage: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const {
    postingMessage,
    postChat,
    messages,
    conversationId,
    setConversationId,
    hasError,
    retryPostChat,
    setCurrentMessageId,
    regenerate,
    getPostedModel,
    loadingConversation,
  } = useChat();

  const { getBotId } = useConversation();
  const { scrollToBottom, scrollToTop } = useScroll();
  const { conversationId: paramConversationId, botId: paramBotId } = useParams();
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [beforeId, setBeforeId] = useState<string | null>(null);
  const socket = io('http://localhost:5000'); // Adjust the URL as needed

  const botId = useMemo(() => {
    return paramBotId ?? getBotId(conversationId);
  }, [conversationId, getBotId, paramBotId]);

  const {
    data: bot,
    error: botError,
    isLoading: isLoadingBot,
    mutate: mutateBot,
  } = useBotSummary(botId ?? undefined);

  const [pageTitle, setPageTitle] = useState('');
  const [isAvailabilityBot, setIsAvailabilityBot] = useState(false);

  useEffect(() => {
    setIsAvailabilityBot(false);
    if (bot) {
      setIsAvailabilityBot(true);
      setPageTitle(bot.title);
    } else {
      setPageTitle(t('bot.label.normalChat'));
    }
    if (botError) {
      if (botError.response?.status === 404) {
        setPageTitle(t('bot.label.notAvailableBot'));
      }
    }
  }, [bot, botError, t]);

  const description = useMemo<string>(() => {
    if (!bot) {
      return '';
    } else if (bot.description === '') {
      return t('bot.label.noDescription');
    } else {
      return bot.description;
    }
  }, [bot, t]);

  const disabledInput = useMemo(() => {
    return botId !== null && !isAvailabilityBot && !isLoadingBot;
  }, [botId, isAvailabilityBot, isLoadingBot]);

  useEffect(() => {
    setConversationId(paramConversationId ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramConversationId]);

  const inputBotParams = useMemo(() => {
    return botId
      ? {
          botId: botId,
          hasKnowledge: bot?.hasKnowledge ?? false,
        }
      : undefined;
  }, [bot?.hasKnowledge, botId]);

  const onSend = useCallback(
    (content: string, base64EncodedImages?: string[]) => {
      postChat({
        content,
        base64EncodedImages,
        bot: inputBotParams,
      });
    },
    [inputBotParams, postChat]
  );

  const onChangeCurrentMessageId = useCallback(
    (messageId: string) => {
      setCurrentMessageId(messageId);
    },
    [setCurrentMessageId]
  );

  const onSubmitEditedContent = useCallback(
    (messageId: string, content: string) => {
      if (hasError) {
        retryPostChat({
          content,
          bot: inputBotParams,
        });
      } else {
        regenerate({
          messageId,
          content,
          bot: inputBotParams,
        });
      }
    },
    [hasError, inputBotParams, regenerate, retryPostChat]
  );

  const onRegenerate = useCallback(() => {
    regenerate({
      bot: inputBotParams,
    });
  }, [inputBotParams, regenerate]);

    useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    } else {
      scrollToTop();
    }
  }, [messages, scrollToBottom, scrollToTop]);

  const { updateMyBotStarred, updateSharedBotStarred } = useBot();
  const onClickBotEdit = useCallback(
    (botId: string) => {
      navigate(`/bot/edit/${botId}`);
    },
    [navigate]
  );

  const onClickStar = useCallback(() => {
    if (!bot) {
      return;
    }
    const isStarred = !bot.isPinned;
    mutateBot(
      produce(bot, (draft) => {
        draft.isPinned = isStarred;
      }),
      {
        revalidate: false,
      }
    );

    try {
      if (bot.owned) {
        updateMyBotStarred(bot.id, isStarred);
      } else {
        updateSharedBotStarred(bot.id, isStarred);
      }
    } finally {
      mutateBot();
    }
  }, [bot, mutateBot, updateMyBotStarred, updateSharedBotStarred]);

  const fetchMessages = async () => {
    const response = await fetch(`/conversation/${conversationId}/messages?before_id=${beforeId}&limit=20`);
    const data = await response.json();
    setMessages(prevMessages => [...data, ...prevMessages]);
    if (data.length > 0) {
      setBeforeId(data[data.length - 1].id);
    }
  };

  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    const handleScroll = async () => {
      if (chatContainer.scrollTop === 0) {
        await fetchMessages();
      }
    };

    chatContainer.addEventListener('scroll', handleScroll);
    return () => chatContainer.removeEventListener('scroll', handleScroll);
  }, [beforeId]);

  const stopGenerating = () => {
    const userId = 'current_user_id';  // Replace with actual user ID
    socket.emit('stop_generation', { user_id: userId });
  };

    return (
    <div onDragOver={onDragOver} onDrop={endDnd} onDragEnd={endDnd}>
      <div className="relative h-14 w-full">
        <div className="flex w-full justify-between">
          <div className="p-2">
            <div className="mr-10 font-bold">{pageTitle}</div>
            <div className="text-xs font-thin text-dark-gray">
              {description}
            </div>
          </div>

          {isAvailabilityBot && (
            <div className="absolute -top-1 right-0 flex h-full items-center">
              <div className="h-full w-5 bg-gradient-to-r from-transparent to-aws-paper"></div>
              <div className="flex items-center bg-aws-paper">
                {bot?.owned && (
                  <StatusSyncBot
                    syncStatus={bot.syncStatus}
                    onClickError={onClickSyncError}
                  />
                )}
                <ButtonIcon onClick={onClickStar}>
                  {bot?.isPinned ? (
                    <PiStarFill className="text-aws-aqua" />
                  ) : (
                    <PiStar />
                  )}
                </ButtonIcon>
                <ButtonPopover className="mx-1" target="bottom-right">
                  {bot?.owned && (
                    <PopoverItem
                      onClick={() => {
                        if (bot) {
                          onClickBotEdit(bot.id);
                        }
                      }}>
                      <PiPencilLine />
                      {t('bot.titleSubmenu.edit')}
                    </PopoverItem>
                  )}
                  {bot?.isPublic && (
                    <PopoverItem
                      onClick={() => {
                        if (bot) {
                          onClickCopyUrl(bot.id);
                        }
                      }}>
                      <PiLink />
                      {copyLabel}
                    </PopoverItem>
                  )}
                </ButtonPopover>
              </div>
            </div>
          )}
        </div>
        {getPostedModel() && (
          <div className="absolute right-2 top-10 text-xs text-dark-gray">
            model: {getPostedModel()}
          </div>
        )}
      </div>
      <hr className="w-full border-t border-gray" />
      <div id="chatContainer" ref={chatContainerRef} style={{ overflowY: 'scroll', height: '500px' }}>
        {messages.length === 0 ? (
          <div className="relative flex w-full justify-center">
            {!loadingConversation && (
              <SwitchBedrockModel className="mt-3 w-min" />
            )}
            <div className="absolute mx-3 my-20 flex items-center justify-center text-4xl font-bold text-gray">
              {!MISTRAL_ENABLED ? t('app.name') : t('app.nameWithoutClaude')}
            </div>
          </div>
        ) : (
          messages.map((message, idx) => (
            <div
              key={idx}
              className={`${
                message.role === 'assistant' ? 'bg-aws-squid-ink/5' : ''
              }`}>
              <ChatMessage
                chatContent={message}
                onChangeMessageId={onChangeCurrentMessageId}
                onSubmit={onSubmitEditedContent}
              />
              <div className="w-full border-b border-aws-squid-ink/10"></div>
            </div>
          ))
        )}
        {hasError && (
          <div className="mb-12 mt-2 flex flex-col items-center">
            <div className="flex items-center font-bold text-red">
              <PiWarningCircleFill className="mr-1 text-2xl" />
              {t('error.answerResponse')}
            </div>

            <Button
              className="mt-2 shadow"
              icon={<PiArrowsCounterClockwise />}
              outlined
              onClick={() => {
                retryPostChat({
                  bot: inputBotParams,
                });
              }}>
              {t('button.resend')}
            </Button>
          </div>
        )}
      </div>

      <div className="absolute bottom-0 z-0 flex w-full flex-col items-center justify-center">
        {bot && bot.syncStatus !== 'SUCCEEDED' && (
          <div className="mb-8 w-1/2">
            <Alert
              severity="warning"
              title={t('bot.alert.sync.incomplete.title')}>
              {t('bot.alert.sync.incomplete.body')}
            </Alert>
          </div>
        )}
        <InputChatContent
          dndMode={dndMode}
          disabledSend={postingMessage}
          disabled={disabledInput}
          placeholder={
            disabledInput
              ? t('bot.label.notAvailableBotInputMessage')
              : undefined
          }
          onSend={onSend}
          onRegenerate={onRegenerate}
        />
        <button id="stopButton" onClick={stopGenerating}>Stop Generating</button>
      </div>
    </div>
  );
};

export default ChatPage;

