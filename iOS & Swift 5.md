Note: learning ios 

# Section 2:

Storyboard -> assistant view -> ViewController

Press control and drag component to link the **IBOutlet** and **IBAction** in ViewController



# Section 4: ViewController, var, let, random

var: for variable

let: for const

Int.random(in: lower ... upper)

...40 41..<81 81...100

a...b a..<b ...b

.randomElement()

Array.shuffle

"a" + "b"

Int, Float, String, Bool



# Section 5: Auto Layout

Size classes and orientation:

Constarints: trailing and leadng: Safe area -> superview

Alignment and pinning:

Containers: editor -> embedded in -> view

StackViews:



# Section 7: Func, sound, link element

func greet(who: String) {}

greet(who: "Yixing")



# Section 8: Control flow

if  x == "green" {} else{}

&& ||  !

switch hardness {

​	case "soft":

​		print(5)

​	default:

​		print("Error")

}

var dict : [String : Int] = ["soft": 5, "hard": 12]

dict["soft"]

Optionals: var x: String? = nil

if x != nil {print(x!)}

progress view



# Section 9:  Structures, design pattern, Model view controller

Struct Town {

​	var list: [String: Int]

​	init(townList: [String: Int]) {

​		self.list = townList

​	}

​	func test(inname outname: String){

​		print(list)

​	}

}

var list=[1,2,3]

Town.list.append(5)  Town.list.count

UIColor

Design Pattern *MVC*: 

Model(Data & Logic)-- request and send --Controller(Mediato)-- send input and modify --View(UI)

func getMilk (money: Int) -> Int {

​	return 1

}

Immutability: func in structure which modify property needs to add mutating



# Section 11: Class, inherit, optional

uisliders

struct pass by value, class pass by reference

class Parent {

​	var health = 100

​	func talk(){print(health}

}

class Child: Parent {

​	var height = 50

​	override func talk(){

​	super.talk()

​	print(height)

​	}

}

Multi-screen: segues



Optional: binding, chaining, nil coalescing:

1. Force unwrapping: optional!

2. Check for nil: 

if optional != nil {optional!}

3. optional binding:

if let safeOptional = optional {safeOptional}

4. Nil coalescing: optional ?? defaultValue
5. Optional chaining: structure: optional?.property



Downcasting: **let** destinationVC = segue.destination **as**! ResultsViewController

Go back to previous: **self**.dismiss(animated: **true**, completion: **nil**)

Color literal



# Section 13: network, json, api, core location

**Sf symbols**

**Dark mode**: 

color: /Assetts/xcassets: Add new color set, Appearance: Any Light Dark

backgroun: drag pdf into , preserve vecotr data + single scale, Appearance: Any Light Dark

**uitextfield**: UITextFieldDelegate, add myTextField.delegate = self in viewDidLoad()

textFieldShouldReturn()textFieldShouldEndEditing()  textFieldDidEndEditing()

**Protocol**: 

Protocol CanFly {

​	func fly()}

Class MyClass: Superclass, FirstProtocol, AnotherProtocol{}

func flyingDemo(obj: CanFly){}

**Delegate design pattern**



**URL**:

// create a url

​    **if** **let** url = URL(string: urlString) {

​      // create a URLSession

​      **let** session = URLSession(configuration: .default)

​      // give the session a task

​      **let** task = session.dataTask(with: url, completionHandler: handle(data:response:error:))

​      // start the task

​      task.resume()

​    }



**Closures**: 

{ (para) -> returnType in

​	statements

}



**JSON**

**func** parseJSON(weatherData: Data) {

​    **let** decoder = JSONDecoder()

​    **do** {

​      **let** decodedData = **try** decoder.decode(WeatherData.**self**, from: weatherData)

​      print(decodedData.main.temp)

​      print(decodedData.weather[0].description)

​    } **catch** {

​      print(error)

​    }

  }



**Computable var**

var property: Int {return 2 + 5}

**Typealiazses**: Decodable Codable

**Name Convention**:

**func** parseJSON(**_** weatherData: Data)     parseJSON(safeData)

**func** performRequest(with urlString: String)     performRequest(with: urlString)



**Extension**

extension Double {

​	func round(to places: Int) -> Double {}

}

Extension to protocol: default func -> optional

extension SomeType: Protocol {}



**Mark** 

//MARK: -



**Core Location**

**import** CoreLocation

requestWhenInUseAuthorization

requestLocation

stopUpdatingLocation



# Section 15: Firebase, Tableview, Cocoapod

**Navigation Controller**:

Select land page, editor, embedded in, navgation

navigationController?.popToRootViewController

navigationItem.hidesBackButton

**Loops**

for i in "123" {}

while true {}

**Cocopods**

pod init

pod install

**Firebase**

**Static Constants**

static let registerSegue = "RegisterToChat"

Constants.registerSegue: do not need to instance

**Table Views**

extension ChatViewController: UITableViewDataSource

UITableViewDataSource: tell how many rows and return cell

cell = tableView.dequeueReusableCell(withIdentifier: K.cellIdentifier, for: indexPath)

UITableViewDelegate: interact with the app

Custom UI

**Type casting**

is -> if human is Animal {} -> type checking

as! -> let fish = animal as! Fish -> forced downcast to subclass

as? -> if let fish = animals[1] as? Fish {}

as -> let animalFish = fish as Animal -> upcast to superclass

**Firebase**

getDocuments

addSnapshotListener: listen for update

Order: 

update UI in closure:  DispatchQueue.main.async {self.messageTextfield.text = ""}

**ViewController Lifecycle**

viewDidLoad() -> viewWillAppear() -> viewDidAppear() -> viewWillDisapper() -> viewDidDisappear()

