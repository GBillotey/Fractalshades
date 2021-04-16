# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from perturbation import Perturbation_mandelbrot
from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
import mpmath
from numpy_utils.xrange import Xrange_array, Xrange_polynomial


def plot():
    """
    Dev
    """
    directory = "/home/geoffroy/Pictures/math/perturb/dev_SAe0001"
    x = "-1.74928893611435556407228"
      #   -1.749288936114352993350644 
          #-1.7492889361143529933506439560953077
    # x = "-1.7492889361143"
         #"-1.749288936114355564073"
    y = "0."
    x = "-1.74928893611435556405228"
    y = "2.e-20"
    
#    x = "-1.74928893611435556401092835373"
#    y = "5.84988579129418726888507089386e-20"
    
    precision = 30
    dx = 5.e-20
    
#    x = "-0.5"
#    y = "0.5"
#    
##    x = "-1.74928893611435556401092835373"
##    y = "5.84988579129418726888507089386e-20"
#    
#    precision = 30
#    dx = "4."
    
#    y = "0."
#    x = "-.5"
##    y = "2."
#    precision = 12
#    dx = 2.e0
    
    
     # emerauld shield
#    x = "-0.7658502608035550"
#    y = "-0.09799552764351510"
#    precision = 18
#    dx = 1.10625e-13
    
    # double embedded J set
#    x = "-1.768667862837488812627419470"# + 0.001645580546820209430325900 i @ 2.7×10-22"
#    y = "-0.001645580546820209430325900"
#    precision = 30
#    dx = "7.5e-22"
    
    # https://fractalforums.org/index.php?action=gallery;sa=view;id=3553
#    x = ("0.2753376477467379935886671248246278815667140698954262859162743630674375101302303013096"
#        "71975356653639860582884204637353849973626635844461696577733396177173659502869597622654"
#        "85804783047336923365261060963100721927003791989610861331863571141065592841226995797739"
#        "723012374298589823921181693139824190379745910243872940870200527114596661654505e+00")
#    y = ("0.6759649405327850670181700456194929502189750234614304846357269137106731032582471677573"
#         "58200829449470582619413145077310704967071714678595763311924422571027117886784050420240"
#         "23624912963178948353210649715186737756302527451352947002166738157907333431349841201085"
#         "24001799351076577642283751627469315124883962453013093853471898311683555782404e-02")
#    precision = 350
#    dx = "3.5e-301"
    
    
        # Eye of the Universe - Mandelbrot Fractal Zoom (e1091) (4k 60fps)
        # https://www.youtube.com/watch?v=pCpLWbHVNhk
        # Zoom: 3.4e1091 / Iterations:  Almost 17 million.
        # Ball method 1 found period: 159413

        # dx = "3.5e-1000"
        # SA running 41200 err:   2.27163164e-159 <<  1.13539426e-153
        # SA stop 41226  5.47729132e-157  1.07442974e-153
        # Trigger hit at 59767
        # dx = "3.5e-1050"
        #  dx = "3.5e-1091"
#    x = ("0.360240443437614363236125244449545308482607807958585750488375814740"
#         "19534605921810031175293672277342639623373172972498773732003537268328"
#         "53176645324012185215795542886617265643241347022999628170292133299808"
#         "95208036363104546639698106204384566555001322985619004717862781192694"
#         "04636274874286301646735457442277944322698262235659413043023245847242"
#         "08166526234929748917304192526511276727824072923155744802070058287745"
#         "66475024380960675386215814315654794021855269375824443853463117354448"
#         "77964709922431184819289397257239866262672525476995097652743127740244"
#         "07528684985887854367053710934424606960907206549089737127599637329148"
#         "49861213100695402602927267843779747314419332179148608587129105289166"
#         "67646129284568573453603369257761849692517057671479669341177679474290"
#         "43334846653016286625329670791747291707141568105305987645252608697312"
#         "33845987202037712637770582084286587072766838497865108477149114659838"
#         "88381879537419515093636998730257437760864962502086429291591337892779"
#         "03440975525919194091373544590975600403748803466375337112719194197231"
#         "35538377394364882968994646845930838049998854075817859391340445151448"
#         "381853615103761584177161812057928")
#    y = ("-0.64131306106480317486037501517930206657949495228230525955617754306"
#         "44485741727536902556370230689681162370740565537072149790106973211105"
#         "27374085199339480328743760623859626228773107599948394046716128884061"
#         "45810912943257099889922691650073943057326832083188346723669475507109"
#         "20088501655704252385244481168836426277052232593412981472237968353661"
#         "47779353033660724773895162581775540106504536227303978833224556734506"
#         "16657567086893592945166682714405252736530837178777012377561442143948"
#         "70245598590883973716531691124286669552803640414068523325276808909040"
#         "31761709268382652150153993239726201201108209872194464311869500122604"
#         "89774300385094701017155554390478847520583348048913896855309461126215"
#         "73416582482926221804767466258346014417934356149837352092608891639072"
#         "74593063936469351321671911452332899069006958867608792365665765602379"
#         "44843247975460242483281565864716626310087413490699614938176001001334"
#         "39721557969263221185095951241491408756751582471307537382827924073746"
#         "76088408170488790204003605661140137878595245210509924249924100320801"
#         "34608784429534086481786923537881537872299402216117310344052035199453"
#         "139116273149008518510721229904925")
#    precision = 2000
#    dx = "1.e-1092"
    
    # The Edge of Infinity  # 5.6e2011
#    x = ("-1.7693831791955150182138472860854737829057472636547514374655282165278881912647564588361634463895296673044858257818203031574874912384217194031282461951137475212550848062085787454772803303225167998662391124184542743017129214423639793169296754394181656831301342622793541423768572435783910849972056869527305207508191441734781061794290699753174911133714351734166117456520272756159178932042908932465102671790878414664628213755990650460738372283470777870306458882898202604001744348908388844962887074505853707095832039410323454920540534378406083202543002080240776000604510883136400112955848408048692373051275999457470473671317598770623174665886582323619043055508383245744667325990917947929662025877792679499645660786033978548337553694613673529685268652251959453874971983533677423356377699336623705491817104771909424891461757868378026419765129606526769522898056684520572284028039883286225342392455089357242793475261134567912757009627599451744942893765395578578179137375672787942139328379364197492987307203001409779081030965660422490200242892023288520510396495370720268688377880981691988243756770625044756604957687314689241825216171368155083773536285069411856763404065046728379696513318216144607821920824027797857625921782413101273331959639628043420017995090636222818019324038366814798438238540927811909247543259203596399903790614916969910733455656494065224399357601105072841234072044886928478250600986666987837467585182504661923879353345164721140166670708133939341595205900643816399988710049682525423837465035288755437535332464750001934325685009025423642056347757530380946799290663403877442547063918905505118152350633031870270153292586262005851702999524577716844595335385805548908126325397736860678083754587744508953038826602270140731059161305854135393230132058326419325267890909463907657787245924319849651660028931472549400310808097453589135197164989941931054546261747594558823583006437970585216728326439804654662779987947232731036794099604937358361568561860539962449610052967074013449293876425609214167615079422980743121960127425155223407999875999884")
#    y = ("0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805538736306127524814939292355930892834392050796724887904921986666045576626946900666103494014904714323725586979789908520656683202658064024115300378826789786394641622035341055102900456305723718684527210377325846307917512628774672005693326232806953822796755832517188873479124361430989485495501124096329421682827330693532171505367455526637382706988583456915684673202462211937384523487065290004627037270912806345336469007546411109669407622004367957958476890043040953462048335322273359167297049252960438077167010004209439515213189081508634843224000870136889065895088138204552309352430462782158649681507477960551795646930149740918234645225076516652086716320503880420325704104486903747569874284714830068830518642293591138468762031036739665945023607640585036218668993884533558262144356760232561099772965480869237201581493393664645179292489229735815054564819560512372223360478737722905493126886183195223860999679112529868068569066269441982065315045621648665342365985555395338571505660132833205426100878993922388367450899066133115360740011553934369094891871075717765803345451791394082587084902236263067329239601457074910340800624575627557843183429032397590197231701822237810014080715216554518295907984283453243435079846068568753674073705720148851912173075170531293303461334037951893251390031841730968751744420455098473808572196768667200405919237414872570568499964117282073597147065847005207507464373602310697663458722994227826891841411512573589860255142210602837087031792012000966856067648730369466249241454455795058209627003734747970517231654418272974375968391462696901395430614200747446035851467531667672250261488790789606038203516466311672579186528473826173569678887596534006782882871835938615860588356076208162301201143845805878804278970005959539875585918686455482194364808816650829446335905975254727342258614604501418057192598810476108766922935775177687770187001388743012888530139038318783958771247007926690")
#    precision = 2200
#    dx = "1.e-2011"
    
    # origin
#    x = "-0.5"# + 0.001645580546820209430325900 i @ 2.7×10-22"
#    y = "-0.0"
#    precision = 15
#    dx = "2.5"
    
    # Light years away
#    Re: -1.7690409012528816824028427681036547222246065136728030061203151986477655139084063011950481722770768927667098134439755159337180516727942814406179951756029928081906
#Im: -0.003160540027255856875974526863608733444435051517757190805697698379351550696026214970411470819014736804857297619204910327207418884876883560019528534608898100264
#Zoom: 5.95426282940E139

# Sun http://www.fractalforums.com/ultra-deep-mandelbrot/sun/
    # Ball method 1 found period: 14584
    # SA stop 65899  2.13334990e-07  2.15581218e-04
    
#    x = ("-0.748204229358803678214332558246869758051087356039445393271164447700613967202248272706368767870104343299544837112533487378306258928435277092927790199523")
#    y = ("-0.086216283574156403625737288880770115522595910290644233472768172845560331986513480840126057262826948897702032801956031059778754857421897015603436264509")
#    precision = 160
#    dx = "3.4e-144"
#Zoom=6e+144
    
    # https://www.deviantart.com/dinkydauset/art/Elephant-spiral-tiling-with-central-transformation-682770711
#    Magnification:
#2^9870
#1.4657408896420228648538645519085e+2971
#
#Coordinates:
#    x = ("-1.7498439088634276800460948042342922498295334226571513188740914298952160154495160835921441755475654893999982369876215851617592098468229564680702922351168965760097475776102438594727651509787077718910539436125381937707617605884622639199601316173201755267877495940801859921008741036365835317535325754579310659698172718927559068739259993943489903422220415937003111035235632482550767932444453657416776260831329826364139257446420512980873151922763831367430838804091679690611736967486334921632295099429770359931163075192348893492407688808513153245277731668770979345334874317532468385607661936063534917048525309601960566949936316979336866617749321597297823714013372646899964899346285783803172833516418670799966711933325159465508213919277200803000055872852120819029687577982703573729797193182111971306136776267365805118136976570306298588531306446692744199554252596954263700577089578723507789675760521447513869517740980613852561229822340340821960377577072823654587832275540464160845654172941864633386678150003773464150372439849205478866686753163400567627899379488621753733939322237710510489899894319695620114081220960799108362857739101378201141015236614994736212399978478462228663563168926677891397591332060282422583766336444493991592761516899516720428428445030542837098451502818031894096231891126049834098808975971389759975985150362816201102121018083628059439863570774916034325209057641891673750907649855902580808823385917908184577563638857597924200360805831466148522376361731640904864693988420015313810216766041336750368569166250692222461154246966278977859304313592575028322631156576324942021305183391989500220481100230016404570236474879345415061504192883601549161673363117250118782954658561767077600417297183475910010828666415715253229639189256362950223239974866161213210866713886300308480188590857334497621417446574144421055762236175453201119570339202660482102753858422401131043468838188246221038734014328251300488439192100059396153720293433575490048680511435309255321529303432822265191296671629725830605376213014511810829167186060245587406244714168349595569030001662916290777088899112276716205440876804982372349280290334339551559023028248022816778418359021526473978858832783517847281048800951679546036790338353953143256548965295380780837896255691843446714391071948040002697770654394052015260255393376175280788923261119479110963358582732044643161872472885492978505953850286721671320296384955791562868032502403710449580790524128525636701394699880064688726954218490244554281678600023139268604824255407806421895967872781732308596780588474743680047670637781940624315346439381802681954581244045582241975959875957962316604643837152815035116622127450845860253963974889814159946447197039023379170539609194628016770286801768992266381824648034095425613913539495478717100958571514945248496902412048555941047782774510767649781516106282821761253790079756292170876360361748187034655726742995944555407678379822885287868173604717194566230010229214603428202446987503287921106362107355612486404563715497")
#    y = ("0.0000000159959473146319642717540099365976948500705065715209798252640461993465538628308271291417567856676375259782625663886020914208896933202641339653886737470769520025797007857355688277340916809887701394149355736780982888336303707426011948432980348233270700327920071207166294721964801722915416135341446136599969054710440421562301919810640516949611055655445374278378523875044932672841451066063600543720534132018364998118432787011667094357767592746119445662386613217689596616495795570879453511631475015465202878892577516514921740593502750764153271780790082960896474165423118429738903206147242877552749256873871105093358252352732331060728483105073241167843748437518392524351874358787417493003841145166331703983481875389753117117932032962953506584753054080076767788009823281476265482712320371045802342552403470963012571639687849320512232492175287872997902858556379366234132626695909943079162218988205311016002657446116689747111210533111481206942309139308952184605691152141960826333395982721345476132751975971022146535204595433352512002578302892490105214373912229371581958063660115836891169373147116580207143925553002261364624545240328038200540148829737033585513962225949452826609947213453364282239248466862885661935973676913754680865468875965089162465135074498164822314446163324863073935405497487064708583916816516533975726680127663202426511499033222075508877083212530649296209358665321153111812420249785544611998025674846336691047595198408371194264939589594429132157351553726174718486809904515040993978014361545147263981626591617347758542367019696332480685643916496305189291127422116306514284473360959979610585543698541400957830806278941724934040493095816324802704019794878310546321765246167763900378695196267121408050678909760576027231496049406396550408670379362262182178544798472264457901620793091459280836757386717166694446359647130402178640031711905205225841807086759410586590898742578188102841587698798631546687215830493331902873137159054492715093065045167280084467604666680883585150536344069812477725101393596817253505556349472358259816524030849996642801406956427167170327092676365024688733492601876964692824759894626974050081365969753000708139172781769560750226186916342763822262989132793630073189830957326040774791732039169455213647452293020342005664871046362252739644996046672804407485968984950340837280695544530862471085846992558012998912101664588802674745392701373309979629845938837179272067628387974487488874154567647656115499120884893301050640923839135312465224721530140263802091571317219010359968405330584283214583937484038726235020015489667750541678639777104491176726074347886713693804393387049866338689190684271772913934340107566951726283444179319632633810736907114231121232694263979534185977644430817648323272004289857100248104618629806930539306672302326896495469553220010205700230257475740272421099410035434715924022827075312970916872859580939642662854699129713633533001625605559242928706357843554207880763875716758293274507690637883513572704426297634947130620506")
#    precision = 3000
#    dx = "0.6822e-2971"
    
    # https://fractalforums.org/share-a-fractal/22/minibrot-at-e1011/4096/msg27413#new
    # Ball method 1 found period: 942775
    # SA stop 738234  9.74681617e-157  7.64821354e-154
# -0.748204229358803678214332558246869758051087356039445393271164447700613967202248272706368767870104343299544837112533487378306258928435277092927790192874716780236963288254400483047288346719618901113243965563394309875788375976678757786312687137043006653947070646389703586769987704327862771473164198096879903235373018273929335379829612316591000469037604247763546965473466633217430290736938579050740437759424242632851404525836664698978088692013173098630731274671233518823383571295988659331621272897683670781689078771047273670746762366645684034982825610290632463657678114589740083310187818084198443962182131076611270807899129666390126366743713456637512585493369578792047829245286587990704199151857003222838116973924591812679163221858308788632664908306213964610796668380094054241586728509347681150498761765493664170078492142536040311063569815415118738055654191686387416289798057058519088598727521823763436468105475345381564303302071014209255689832840138280428968143315421244096260954201808625890826750398305496469405913813658450874907613717
# 0.0862162835741564036257372888807701155225959102906442334727681728455603319865134808401260572628269488977020328019560310597787548574218970156034362718663172676802590633649051872006782504900598829186418318571082348873484485300866243791697456635945986039445803832090925532369884563615510752850461365785649506641632496945646579442564229883749898535222733599042548992895481349521811154250675704588023013710306712562930121169737776968069102802059349061373842932739452135672841852798937147879235136930212578363927360750005606224712490851613661528011271585030721837978121897603558631651653221772156591627520721420187801959810531446002479703400207748882520441215642596841992485850702213156022538690590294821446474103779041220606288453195383896603789866826370833553338024755380610405849855803783132963934673267664028346948564415882082909043848944314528516027312019777341023394259780858292128125096661992698290818317102937023491582493886164693435739012451354746076124583938051110841477497793448983971222377922461953323811198617102528712945435615
    x = "-0.748204229358803678214332558246869758051087356039445393271164447700613967202248272706368767870104343299544837112533487378306258928435277092927790192874716780236963288254400483047288346719618901113243965563394309875788375976678757786312687137043006653947070646389703586769987704327862771473164198096879903235373018273929335379829612316591000469037604247763546965473466633217430290736938579050740437759424242632851404525836664698978088692013173098630731274671233518823383571295988659331621272897683670781689078771047273670746762366645684034982825610290632463657678114589740083310187818084198443962182131076611270807899129666390126366743713456637512585493369578792047829245286587990704199151857003222838116973924591812679163221858308788632664908306213964610796668380094054241586728509347681150498761765493664170078492142536040311063569815415118738055654191686387416289798057058519088598727521823763436468105475345381564303302071014209255689832840138280428968143315421244096260954201808625890826750398305496469405913813658450874907613717"
    y = "0.0862162835741564036257372888807701155225959102906442334727681728455603319865134808401260572628269488977020328019560310597787548574218970156034362718663172676802590633649051872006782504900598829186418318571082348873484485300866243791697456635945986039445803832090925532369884563615510752850461365785649506641632496945646579442564229883749898535222733599042548992895481349521811154250675704588023013710306712562930121169737776968069102802059349061373842932739452135672841852798937147879235136930212578363927360750005606224712490851613661528011271585030721837978121897603558631651653221772156591627520721420187801959810531446002479703400207748882520441215642596841992485850702213156022538690590294821446474103779041220606288453195383896603789866826370833553338024755380610405849855803783132963934673267664028346948564415882082909043848944314528516027312019777341023394259780858292128125096661992698290818317102937023491582493886164693435739012451354746076124583938051110841477497793448983971222377922461953323811198617102528712945435615"
    dx = "1.e-1000"
    precision = 1100
    nx = 1600
    xy_ratio = 1. #/ 1.618
#    xy_ratio = 1.
    theta_deg = 0.
    complex_type = np.complex128#("Xrange", np.complex128)
    # complex_type = ("Xrange", np.complex64)
#    complex_type = (np.complex64)
    
#    max_iter = 10000
#    M_divergence = 1.e3
#    epsilon_stationnary = 1.e-3
#    pc_threshold = 0.2

    mandelbrot = Perturbation_mandelbrot(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type, projection="cartesian",
                 precision=precision)
#    print("1")
    mandelbrot.full_loop(
            file_prefix="dev",
            subset=None,
            max_iter=5000000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params={"cutdeg": 64},
            glitch_eps=1.e-3,
            interior_detect=False)
#    print("2")

    stationnary = Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")
    glitched = Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x >= 2), mode="r+raw")
#    print("3")

#(self, file_prefix, subset, max_iter, M_divergence,
#                   epsilon_stationnary, pc_threshold=1.0, iref=0, 
#                   SA_params=None, glitch_eps=1.e-3):
#    c0 = mandelbrot.x + 1j * mandelbrot.y
#    order = mandelbrot.ball_method(c0, dx/nx * 10., 100000)
##    if order is None:
##        raise ValueError()
#    order = None
#    print("order", order)
#    newton_cv, nucleus = mandelbrot.find_nucleus(c0, order)
##    print("nucleus", newton_cv, nucleus)
##    print("shift", c0 - nucleus)
#    shift = c0 - nucleus
#    if (abs(shift.real) < mandelbrot.dx) and (abs(shift.imag) < mandelbrot.dy):
#        print("reference nucleus found at:\n", nucleus, order)
#        print("img coords:\n",
#              shift.real / mandelbrot.dx,
#              shift.imag / mandelbrot.dy)
#    else:
#        raise ValueError()
#    
#    z = mpmath.mp.zero
#    dzdc = mpmath.mp.zero
#    d2zdc2 = mpmath.mp.zero
#    z_tab = [dzdc]
#    dzdc_tab = [d2zdc2]
#    for i in range(1, order + 30):
#        dzdc = 2 * dzdc * z + 1.
#        z = z**2 + c0
#        z_tab += [z]
#        dzdc_tab +=[dzdc]
#        if   (i >= order):
#            print("z", i, z_tab[i], z_tab[i-order] - z_tab[i])
    

#
#    mandelbrot.dev_loop(
#        file_prefix="dev",
#        subset=None,
#        max_iter=max_iter,
#        M_divergence=M_divergence,
#        epsilon_stationnary=epsilon_stationnary,
#        pc_threshold=pc_threshold)
    potential_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    base_data_key = ("DEM_explore",
            {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
    
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([1, 0, 1]) / 255.
    black_green = np.array([3., 54., 3.]) / 255.
    black_blue = np.array([0, 0., 75]) / 255.
    brown = np.array([50., 0., 0.]) / 255.
    #black = np.array([0, 25., 50]) / 255.
    brown_black = np.array([75, 0, 0]) / 255.
    brown_bblack = np.array([35, 0, 0]) / 255.
    true_white = np.array([250, 250, 250]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    citrus = np.array([222, 223, 129]) / 255.
    citrus2 = np.array([103, 189, 0]) / 255.
    citrus_white = np.array([252, 251, 226]) / 255.
    gold2 = np.array([244, 216, 145]) / 255.
    sand = np.array([240, 159, 88]) / 255.
    copper = np.array([222, 145, 117]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    navy_blue = np.array([0, 19, 95]) / 255.
    sea_blue = np.array([33, 153, 147]) / 255.
    mint_green = np.array([112, 214, 203]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    orchid1 = np.array([157, 37, 97]) / 255.
    orchid2 = np.array([326, 222, 129]) / 255.
    old_rose3 = np.array([181, 83, 142]) / 255.
    old_rose = np.array([245, 30, 42]) / 255.
    old_rose2 = np.array([238, 47, 108]) / 255.
    pomelo1 = np.array([235, 124, 102]) / 255.
    pomelo2 = np.array([130, 13, 52]) / 255.
    #old_rose = np.array([255, 0, 100]) / 255.
    cream = np.array([224, 217, 202]) / 255.
    cream2 = np.array([197, 232, 178]) / 255.
    wheat1 = np.array([244, 235, 158]) / 255.
    wheat2 = np.array([246, 207, 106]) / 255.
    wheat3 = np.array([191, 156, 96]) / 255.
    lavender1 = np.array([154, 121, 144]) / 255.
    lavender2 = np.array([140, 94, 134]) / 255.
    lavender3 = np.array([18, 16, 58]) / 255.
    #cream2 = np.array([238, 231, 50]) / 255.
    
    #black_blue = sand
    
#    royalblue = sea_blue
#    deepskyblue = sea_blue
#    navy_blue = black

#    color_gradienta2 = Color_tools.Lch_gradient(purple, old_rose,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradienta1c = Color_tools.Lch_gradient(old_rose, gold,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradienta1b = Color_tools.Lch_gradient(gold, brown_black,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradienta1 = Color_tools.Lch_gradient(brown_black, royalblue,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradienta0 = Color_tools.Lch_gradient(royalblue, old_rose2,  200,
#                                              f= lambda x:x) #**1.)
#
#    color_gradient0 = Color_tools.Lch_gradient(old_rose2, copper,  200,
#                                              f= lambda x:x) #**1.)
    def wave(x):
        return 0.5 + (0.4 * (x - 0.5) - 0.6 * 0.5 * np.cos(x * np.pi * 3.))
        
    color_gradient1 = Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x))#x**2 * (3. - 2.*x))
    color_gradient2 = Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient3 = Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x))
    color_gradient4 = Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient5 = Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient6 = Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient7 = Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient8 = Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x)) #**1.)
    
    color_gradient9 = Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x)) #**1.)
    
    color_gradient10 = Color_tools.Lch_gradient(wheat3, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient11 = Color_tools.Lch_gradient(wheat1, lavender2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient12 = Color_tools.Lch_gradient(lavender2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient13 = Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    
    
    color_gradient14 = Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x))
    color_gradient15 = Color_tools.Lch_gradient(wheat3, wheat1, 200,
                                              f= lambda x: wave(x))
    color_gradient16 = Color_tools.Lch_gradient(wheat1, lavender1, 200,
                                              f=  lambda x: wave(x))
    color_gradient17 = Color_tools.Lch_gradient(lavender1, wheat1, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient18 = Color_tools.Lch_gradient(wheat1, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient19 = Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient20 = Color_tools.Lch_gradient(lavender2, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient21 = Color_tools.Lch_gradient(lavender3, lavender1, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient22 = Color_tools.Lch_gradient(lavender1, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient23 = Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient24 = Color_tools.Lch_gradient(lavender2, citrus2, 200,
                                              f= lambda x: wave(x)) #**1.)
    
#    color_gradient1cc = Color_tools.Lch_gradient(mint_green, navy_blue,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradient2c = Color_tools.Lch_gradient(navy_blue, royalblue,  200,
#                                              f= lambda x:x) #**1.)
#    color_gradient3c = Color_tools.Lch_gradient(royalblue, copper,  200,
#                                              f= lambda x:x) #**1.)
    
    colormap = (-Fractal_colormap(color_gradient8) -
                Fractal_colormap(color_gradient7) - 
                Fractal_colormap(color_gradient6) - 
                Fractal_colormap(color_gradient5) -
                Fractal_colormap(color_gradient4) -
                Fractal_colormap(color_gradient3) - 
                Fractal_colormap(color_gradient2) - 
                Fractal_colormap(color_gradient1)) # + 
    colormap = (Fractal_colormap(color_gradient4)+#, plt.get_cmap("magma")) + Fractal_colormap((0.1, 1.0, 200), plt.get_cmap("magma"))
                Fractal_colormap(color_gradient5) + 
                Fractal_colormap(color_gradient6) + 
                Fractal_colormap(color_gradient7) + 
                Fractal_colormap(color_gradient8) + 
                Fractal_colormap(color_gradient9) + 
                Fractal_colormap(color_gradient10) + 
                Fractal_colormap(color_gradient11) + 
                Fractal_colormap(color_gradient12) + 
                Fractal_colormap(color_gradient13) + 
                Fractal_colormap(color_gradient14) + 
                Fractal_colormap(color_gradient15) + 
                Fractal_colormap(color_gradient16) + 
                Fractal_colormap(color_gradient17) + 
                Fractal_colormap(color_gradient18) + 
                Fractal_colormap(color_gradient19) + 
                Fractal_colormap(color_gradient20) + 
                Fractal_colormap(color_gradient21) + 
                Fractal_colormap(color_gradient22) + 
                Fractal_colormap(color_gradient23) + 
                Fractal_colormap(color_gradient24) )

    colormap.extent = "repeat"
    colormap2 = Fractal_colormap(color_gradient2)
    phi = 0.60# 0.20
    k = 500.
    plotter = Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=potential_data_key,
        base_data_prefix="dev",
        base_data_function=lambda x:x,# np.sin(x*0.0001),
        colormap=colormap,
        probes_val=np.linspace(0., 1., 22) * 428  - 00.,#[0., 0.5, 1.], #phi * k * 2. + k * np.array([0., 1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 3.5,
        probes_kind="z",
        mask=glitched)
    
    
    plotter.add_calculation_layer(postproc_key=potential_data_key)
    
    layer1_key = ("DEM_shade", {"kind": "potential",
                                "theta_LS": 45.,
                                "phi_LS": 33.,
                                "shininess": 4.,
                                "ratio_specular": 2.})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
                        # disp_layer=True,
                         normalized=False, hardness=1.5,  
            skewness=0.15, shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})
    
#    layer2_key = ("field_lines", {})
#    Fourrier = ([1., 0., 0., 0.], [0., 0., 0., 0.])
#    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
#                         hardness=0.5, intensity=0.38,
#                         blur_ranges=[],#[[0.99, 0.99, 1.0]], 
#                         shade_type={"Lch": 0., "overlay": 2., "pegtop": 0.}) 
#
#    layer_key = ("DEM_explore",
#        {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
#    plotter.add_grey_layer(
#        postproc_key=layer_key,
#        intensity=0.5, 
#        normalized=False,
#        skewness=0.0, 
#        shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})
#
    plotter.plot("dev", mask_color=(1., 0., 0.))

if __name__ == "__main__":
    plot()