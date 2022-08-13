from datasets import load_from_disk
import pandas as pd


def data_in_black_list(train_texts):
    black_list=[
        """17.\nחבריי ואני שיחקנו מתנקש nerf (בחינם טווח). היה לי אקדח חנון רוסס בשחור. אני badass המזוין. הייתי מצמרר בבית החברים שלי בחלק מוצל גבולי של העיר. התקשרו אליי שומה שהייתה לי בצוות החנון האחר שהם היו שולחים חוליה לתקוף אותי. לא משנה מה, עדיין יש לי עשר דקות. 20 דקות מאוחר יותר חברים כמו שאנחנו צריכים לעזוב. חרא, כן. ללכת בחוץ (בשעתי הלילה זה של אגב) SUV השחור מתגנבת לאט מאחור. לא טיול עסקה גדול מהר יותר. מכונית מאיצה. לטבול להגדיר. לקפוץ גדר ברזל יצוק (ולקרוע את הסווטשרט BTBAM) פועל דרך חצרות האחוריות כדי להגיע למכונית שלי. הם מצאו אותו יש לו סיכן החוצה. an't לחזור. קורא למ"כ שלי. להיות שם ב20; על פרופיל נמוך. 10-4. למצוא דרך סמטה נחמדה. אורות תנועת גילוי ללכת. לא ביגי. צ'יל 15. המ"כ קורא. בנקודת התמצית. סילון. מכונית נעצרת מאחוריו 10 שניות מאוחר יותר. גבר קופץ החוצה מכונית. אני רץ למנהיג נבחרת. אדם רואה אקדח חנון שחור. שולף את מה שנראה כמו נשק m9ish (אני לא יודע בנשק). אתה דפוק לא הולך לשום מקום. יושב לי ולחברים שלי על מרפסת ואומר האזעקה בסטודיו לריקוד שאשתו יצאה לדרך. מאחוריי לחתום אומר <wife's לרקוד studio> אופס. החברים שלי ואמרו שהוא לא יכול לעצור אותנו. אחי יש אקדח הצביע עלינו. הוא יכול לעכב אותנו. לשכנע אותו ששחקנו במשחק. הוא רואה את בני נוער. שואל מה תיכון אנחנו הולכים. <Insert גבוה School>. חרא. אני Pres של אלום אלומיניום ב<Insert גבוה School>. איזה חרא. להתחיל לדבר. הוא יודע שהורים של מנהיגי כיתה. שואל אותי את שם המשפחה שלי. גלו שהוא הלך לבית ספר יסודי, עם הדודה שלי במשך 8 שנים. מה לעזאזל. השאר על פתק טוב. עדיין באזור. המ"כ מושך יותר. אומר לנו לחכות במכוניתו בעת שהוא מקבל קצת גלידה. Duh לעזאזל, מאנג. בואו נלך. מספיק חרא שקרה הלילה. אא גוצ'י. חבר ואני לא מקבלים את. ישב במכונית כאשר חולית אויב מזהה אותנו. לעזאזל. לנעול את הדלתות. 5 אנשים מקיפים את הרכב. תנסה להסביר מה בדיוק קרה וזה שנסיים להלילה. lolno, סיפור נחמד. נלכד במכונית ל45 דקות נוספות עד שהמ"כ שלנו יכול לדבר עם מפקד הכיתה שלהם קוראים לשעות 12. הפסקת אש.""",
        """Seems like this relationship is pretty rocky, honestly. \n She doesn't sound like a good person to date. Now, not everything you listed is awful, but I can understand that in context of your stressful relationship, how you can perceive it as an issue. \n That's what dating in your teens is all about- it's about learning relationship skills, and one of them is learning when someone's not a fit for you. \n Typically, when issues arise in your relationship, your best bet is to communicate to your partner what you perceive the problem as, and try to find some sort of solution (together). While I'm not saying that you shouldn't discuss these matters with her (you absolutely should!), recognize that not all problems can be solved, and sometimes those unsolvable problems are deal breakers. \n \n Here are some reasons why I think you're best leaving your current relationship: \n \n You guys clearly need different levels of affection.  ^While ^it's ^possible ^to ^work ^on ^this, ^it's ^a ^pretty ^big ^deal ^breaker ^to ^lots ^of ^folks. ^I'm ^not ^really ^into ^"I ^love ^you"s ^or ^cuddling, ^and ^that's ^stuff ^that ^a ^lot ^of ^people ^can't ^accept, ^and ^it's ^difficult ^for ^both ^parties ^to ^compromise. \n You're uncomfortable with how she treats your friends.  ^Maybe ^she's ^trying ^really ^hard ^to ^impress ^your ^friends, ^because, ^y'know, ^they're ^your ^friends ^(and ^thus, ^perhaps ^she ^thinks ^it'll ^impress ^you?). ^Or ^maybe ^she's ^flirting ^with ^them. \n She's very insecure about you talking to other girls.  ^Not ^much ^to ^say. ^This ^is ^a ^sort ^of ^a ^red ^flag. ^Trust ^issues ^in ^relationships ^are ^never ^a ^good ^sign. \n Coincidentally, you're very insecure about her talking to other guys.  ^Not ^saying ^that ^it's ^totally ^unwarranted, ^but ^it ^is ^an ^issue ^(and ^a ^little ^bit ^much).  \n Amount of years doesn't correlate to quality.  ^A ^year ^of ^one's ^life ^does ^seem ^like ^an ^investment, ^and ^I ^imagine ^in ^high ^school ^especially, ^but ^it ^shouldn't ^be ^what ^holds ^a ^relationship ^together. \n Breaking things off is a totally viable option.  ^Now, ^I'm ^not ^saying ^that ^you ^should ^dump ^someone ^over ^any ^little ^thing, ^heck, ^if ^it ^was ^just ^one ^or ^two ^of ^these ^problems ^you ^could ^probably ^work ^through ^it- ^but ^that's ^not ^always ^the ^case ^(like ^this ^one).  \n \n \n It's a little plain to see, especially after compiling this stuff, that you guys aren't really a fit. There are tonnes of reasons why, maybe she's only acting this way because she  wants  you to break up with her (a lot of those "how to lose a guy" behaviors goin' on), maybe she isn't really ready for this relationship (explains why she's having a hard time prioritizing you, and instead spending time with other single guys), maybe she's bored of her current relationship (and coping with it in some pretty hurtful ways), heck, maybe in the off chance, she's not even monogamous and just doesn't know it yet (even polyamorous/non-monogamous folks have insecurities with their partners- and I imagine moreso at 15). Or it could be something else entirely.""",
        ]
    # This function mainly works in train_texts 
    df = pd.DataFrame(train_texts)
    for k in df:
        if df[k] in pd.Series(black_list):
            print('training data contains sample in black list')
            print(df[k])
            exit()
        if k in [22016, 22023, 29160, 29161, 40576, 96624]:
            print(k + ': ' + df[k])
        
if __name__ == '__main__':
    dataset = load_from_disk("../../reddit_clean")
    train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
    val_texts, val_labels = dataset['valid']['content'], dataset['valid']['summary']
    test_texts, test_labels = dataset['test']['content'], dataset['test']['summary']

    # All the test here 
    data_in_black_list(train_texts)
    # Upload to hub after passing all tests
    dataset.push_to_hub("SophieTr/reddit_clean")
